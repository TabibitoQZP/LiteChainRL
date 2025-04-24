import os
import ray
import simplejson as json
from ray.util.queue import Queue
from uuid import uuid4
from datetime import datetime

from rollout.lora_engine import LoRAEngine
from reward.base_reward import BaseReward
from trainer import Logger


def save_outupt(data, root="data/rollout"):
    os.makedirs(root, exist_ok=True)
    uid = str(uuid4())
    timestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(root, f"({timestr}){uid}.json")
    with open(path, "w") as js:
        json.dump(data, js, indent=2)


@ray.remote
class LoRAEngineWorker:
    def __init__(self, engine_config, inner_reward_class):
        self.engine_config = engine_config
        self.lora_engine = LoRAEngine(engine_config)
        self.reward_instance = inner_reward_class(self.lora_engine)

    def listen(self, in_q, out_q):
        while True:
            requests = in_q.get()
            if requests == -1:
                break
            responses = self.lora_engine.multi_turn_gen(requests[0], requests[1])
            responses = self.reward_instance.reward(responses, requests[2])
            out_q.put(responses)

    def update_weight(self, lora_path=None):
        if lora_path:
            self.lora_engine.update_lora(lora_path)
        else:
            self.lora_engine.update_lora(self.engine_config.lora_path)

    def ready(self):
        return True


@ray.remote
class RolloutManager:
    def __init__(
        self,
        dataset,
        inner_reward_class,
        devices_list,
        engine_config,
        log_path,
    ):
        self.logger = Logger(os.path.join(log_path, "rollout_log.jsonl"))
        self.rollout_output_path = os.path.join(log_path, "rollout")
        self.index = self.logger.read_last().get("index", 0)
        print(f"Rollout will start at {self.index}.")
        self.dataset = dataset

        self.workers = []
        for devices in devices_list:
            engine_config.cuda_visible_devices = devices
            engine_config.seed += 1
            self.workers.append(
                LoRAEngineWorker.remote(engine_config, inner_reward_class)
            )

        self.in_q = Queue()
        self.out_q = Queue()

    def start_a_rollout(
        self,
        sampling_size,
        update_batch,
        out_batch,
        out_queue,
    ):
        assert update_batch % sampling_size == 0, (
            f"update_batch({update_batch}) should be devided by sampling_size({sampling_size})."
        )
        send_batch = update_batch // sampling_size
        # send data to the input queue
        for idx in range(send_batch):
            if self.index >= len(self.dataset):
                return True
            prompt, env, metadatum = self.dataset[self.index]
            self.index += 1
            prompts = [prompt for _ in range(sampling_size)]
            envs = [env.copy() for _ in range(sampling_size)]
            metadata = [metadatum for _ in range(sampling_size)]
            self.in_q.put((prompts, envs, metadata))

        # send stop signals to the input queue
        for w in self.workers:
            self.in_q.put(-1)

        for w in self.workers:
            w.listen.remote(self.in_q, self.out_q)

        # get items
        all_items = []
        for idx in range(send_batch):
            all_items.extend(self.out_q.get())
            while len(all_items) >= out_batch:
                save_outupt(all_items[:out_batch], self.rollout_output_path)
                out_queue.put(all_items[:out_batch])
                all_items = all_items[out_batch:]

        return False

    def update_weight(self, lora_path=None):
        # add next batch index for reload
        self.logger.append({"index": self.index})
        for w in self.workers:
            w.update_weight.remote(lora_path)
        for w in self.workers:
            ray.get(w.ready.remote())
        return True
