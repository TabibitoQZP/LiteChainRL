import random
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from trigger.code_trigger import CodeTrigger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from reward.toy_reward import ToyReward

# config
model_path = "/mnt/data/litechainrl/models/Qwen2.5-Coder-32B-Instruct/"
lora_path = "./data/lora/"

prompt_template = """
You will be given a list of integers and a result integer. You need to use every element once and only once with the oprand `+` and `-` to generate a equation that equals the result integer. The following is an example:

**Integer List**: [19, 36, 55, 7]

**Result Integer**: 65

<Answer>55 + 36 - 7 -19</Answer>

You can use python code to help you find the combination. Specifically, you can write your python code wrapped in ```python```. You need to use **print** function to print the result you want. When you figure out the combination. You should write the result wrapped in <Answer></Answer>. You only need to list the combination and do not contain equal and result integer in it. The following is your task.

**Integer List**: {integer_list}

**Result Integer**: {result_integer}

Now start your analysis step by step. Try to use python code and give your final result wrapped in <Answer></Answer>.
"""


class ToyDataset(Dataset):
    def __init__(
        self,
        model=model_path,
        minval=1,
        maxval=128,
        mincount=4,
        maxcount=8,
        dataset_size=16384,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.minval = minval
        self.maxval = maxval
        self.mincount = mincount
        self.maxcount = maxcount
        self.dataset_size = dataset_size

        self.oprands = "+ -".split()

        tokenizer = AutoTokenizer.from_pretrained(model)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt_template.strip(),
            },
        ]
        self.prompt_template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        self.raw_dataset = self._init_dataset()

    def _single_gen(self):
        count = random.randint(self.mincount, self.maxcount)
        values = [random.randint(self.minval, self.maxval) for _ in range(count)]

        equation_string = str(values[0])
        for item in values[1:]:
            equation_string += random.choice(self.oprands) + str(item)
        result = eval(equation_string)
        random.shuffle(values)
        return values, result

    def _init_dataset(self):
        raw_dataset = []
        for _ in tqdm(range(self.dataset_size), "init toy dataset..."):
            raw_dataset.append(self._single_gen())
        return raw_dataset

    def __getitem__(self, idx):
        values, result = self.raw_dataset[idx]

        prompt = self.prompt_template.format(
            integer_list=str(values),
            result_integer=str(result),
        )

        trigger = CodeTrigger({})
        metadata = {"values": values, "result": result}

        return prompt, trigger, metadata

    def __len__(self):
        return self.dataset_size


def process_to_batch(tokenizer, data, max_model_len, device):
    batch = []
    mask = []
    ref_logprobs = []
    rollout_logprobs = []
    reward = []
    for d in data:
        batch.append([item["token_id"] for item in d["token_info"]])
        mask.append([item["mask"] for item in d["token_info"]])
        ref_logprobs.append(
            [item["ref_logprob"] for item in d["token_info"][1:]]
            + [0 for _ in range(max_model_len - len(d["token_info"]))]
        )
        rollout_logprobs.append(
            [item["rollout_logprob"] for item in d["token_info"][1:]]
            + [0 for _ in range(max_model_len - len(d["token_info"]))]
        )
        reward.append([d["reward"]])

    # 这个获得的是attention mask
    batch_sentences = {
        "input_ids": batch,
    }
    padded = tokenizer.pad(
        batch_sentences,
        padding="max_length",
        max_length=max_model_len,
    )
    batch = padded["input_ids"]
    attn_mask = padded["attention_mask"]

    # 这个获得的是训练的mask, 用于交叉熵损失
    batch_sentences["attention_mask"] = mask
    padded = tokenizer.pad(
        batch_sentences,
        padding="max_length",
        max_length=max_model_len,
    )
    training_mask = padded["attention_mask"]

    batch = torch.tensor(batch, dtype=torch.int64).to(device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.bool).to(device)
    training_mask = torch.tensor(training_mask, dtype=torch.bool).to(device)
    ref_logprobs = torch.tensor(ref_logprobs, dtype=torch.bfloat16).to(device)
    rollout_logprobs = torch.tensor(rollout_logprobs, dtype=torch.bfloat16).to(device)
    reward = torch.tensor(reward, dtype=torch.bfloat16).to(device)

    return (
        batch,
        attn_mask,
        training_mask[:, 1:],
        ref_logprobs,
        rollout_logprobs,
        reward,
    )


if __name__ == "__main__":
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        update_event = mp.Event()
        mini_batch_queue = mp.Queue()
        from rollout.engine_worker import engine_manager

        manag_proc = mp.Process(
            target=engine_manager,
            kwargs={
                "dataset": ToyDataset,
                "devices_list": ["7", "6"],
                "group_size": 64,
                "update_batch": 256,
                "update_event": update_event,
                "mini_batch": 4,
                "mini_batch_queue": mini_batch_queue,
                "reward_class": ToyReward,
                "model": model_path,
                "lora_path": lora_path,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.5,
                "max_token_per_turn": 1024,
                "seed": 42,
                "qlora": False,
            },
        )

        manag_proc.start()
