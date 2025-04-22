import os
import re
import random
import pandas as pd
import ray
from ray.util.queue import Queue
import simplejson as json
from torch.utils.data import Dataset

from trigger.code_trigger import CodeTrigger
from rollout.engine_worker import RolloutManager
from rollout.lora_engine import LoRAEngineConfig
from reward.code_eval_reward import CodeEvalReward
from trainer.trainer_worker import TrainerWorker


class LoadWikiDB:
    def __init__(
        self,
        sqlCreationPath,
        setsPath,
        dbRoot,
        shuffle=True,
    ):
        self.dbRoot = dbRoot
        with open(sqlCreationPath, "r") as js:
            self.sqlCreation = json.load(js)
        with open(setsPath, "r") as js:
            self.sets = json.load(js)

        if shuffle:
            for item in self.sets:
                random.shuffle(item)
        self.maxTableLen = len(self.sets)  # except it

        self.pattern = re.compile(
            r"""
            FOREIGN\s+KEY
            \s*\(\s*"(?P<column>[^"]+)"\s*\)
            \s+REFERENCES\s*
            "(?P<ref_table>[^"]+)"\s*
            \(\s*"(?P<ref_column>[^"]+)"\s*\)
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def extract_foreign(self, sql_query):
        matches = list(self.pattern.finditer(sql_query))
        results = []
        for m in matches:
            results.append(m.groupdict())
        return results

    def readItem(self, tableSz, itemIdx):
        assert tableSz < self.maxTableLen, "tableSz out of range"
        assert itemIdx < len(self.sets[tableSz]), "itemIdx out of range"

        item = self.sets[tableSz][itemIdx]
        forein_references = {}
        tables = {}
        env = {}
        for tn in item["set"]:
            forein_references[tn] = self.extract_foreign(
                self.sqlCreation[item["database"]][tn]
            )
            tables[tn] = env[tn] = pd.read_csv(
                os.path.join(self.dbRoot, item["database"], "tables", f"{tn}.csv")
            )
        exec("import pandas as pd\nimport numpy as np", env)

        return forein_references, tables, env

    def getMap(self, s, e):
        mapList = []
        for i in range(s, e):
            for j in range(len(self.sets[i])):
                mapList.append((i, j))
        return mapList


prompt_template = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You will be given some tables saved in DataFrame. These tables are from a database and have the following foreign key relations.

```
{foreign}
```

You can write Python codes to explore a database. The following is an example.

```python
{python_code}
```

<Code Result>
{outputs}
</Code Result>

Once you write a Python code block wrapped in '```python```'. It will be executed automatically and the printed results will be give back wrapped in '<Code Result></Code Result>'. Try to use this feature to explore the database and find an interesting conclusion. Your final conclusion should be wrapped in '<Answer></Answer>'.<|im_end|>
<|im_start|>assistant
"""


class WikiDBDataset(Dataset):
    def __init__(
        self,
        sqlCreationPath,
        setsPath,
        dbRoot,
        start,
        end,
        promptTemplate,
    ):
        self.lwd = LoadWikiDB(sqlCreationPath, setsPath, dbRoot)
        self.mapList = self.lwd.getMap(start, end)
        self.promptTemplate = promptTemplate

    def __len__(self):
        return len(self.mapList)

    def __getitem__(self, index):
        realIdx = self.mapList[index]
        forein_references, tables, env = self.lwd.readItem(realIdx[0], realIdx[1])

        foreign = json.dumps(forein_references, indent=2)

        code_list = []
        result_list = []
        for k, v in tables.items():
            code_list.append(f"print('# {k}')\nprint({k}.convert_dtypes().dtypes)")
            result_list.append(f"# {k}\n{v.convert_dtypes().dtypes}")

        prompt = self.promptTemplate.format(
            foreign=foreign,
            python_code="\n".join(code_list),
            outputs="\n".join(result_list),
        )
        return prompt, CodeTrigger(env), None


def init_lora(modelPath, loraPath, peftConfig=None):
    import peft
    from transformers import AutoModelForCausalLM

    if peftConfig is None:
        # 默认的config就是r=8/16, alpha=r/2r
        peftConfig = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    model = AutoModelForCausalLM.from_pretrained(modelPath).to("cpu")

    os.makedirs(loraPath, exist_ok=True)
    model.add_adapter(peftConfig, adapter_name="init_lora")
    model.save_pretrained(loraPath)


if __name__ == "__main__":
    ray.init()

    # init dataset
    sqlCreationPath = "./data/SQLCreation.json"
    setsPath = "./data/sets.json"
    dbRoot = "/mnt/data/litechainrl/dataset/WikiDBs/part-0/"
    start = 4
    end = 32
    dataset = WikiDBDataset(
        sqlCreationPath,
        setsPath,
        dbRoot,
        start,
        end,
        prompt_template,
    )

    # config model
    modelPath = "/mnt/data/litechainrl/models/Qwen2.5-Coder-7B-Instruct/"
    loraPath = "data/lora"
    max_model_len = 8192
    rollout_device_list = "7 6".split()
    engine_config = LoRAEngineConfig(modelPath, loraPath, qlora=True)

    # config traning
    sampling_batch = 8
    mini_batch = 2
    train_batch_size = 16
    update_batch = 32
    deepspeed_devices = "4,5"

    # config deepspeed
    ds_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": mini_batch,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "cpu"},
        },
    }

    # config grpo
    epsilon = 0.2
    beta = 0.8

    # init lora
    init_lora(modelPath, loraPath)

    # init engine manager
    out_queue = Queue()
    rollout_manager = RolloutManager.remote(
        dataset,
        CodeEvalReward,
        ["7", "6"],
        engine_config,
    )

    # init trainer
    trainer_worker = TrainerWorker.options(resources={"GPU4": 1, "GPU5": 1}).remote(
        modelPath, loraPath, ds_config, deepspeed_devices
    )

    while True:
        rollout_manager.start_a_rollout.remote(
            sampling_batch,
            update_batch,
            mini_batch,
            out_queue,
        )
        ray.get(
            trainer_worker.step.remote(
                update_batch,
                out_queue,
                max_model_len,
                epsilon,
                beta,
                sampling_batch,
            )
        )
        rollout_manager.update_weight.remote()
        break
