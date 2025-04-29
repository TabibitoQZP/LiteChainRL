import os
import ray
import argparse
import json

from ray.util.queue import Queue

from trainer import init_lora
from rollout.lora_engine import LoRAEngineConfig
from rollout.engine_worker import RolloutManager
from trainer.trainer_worker import TrainerManager

# NOTE: Please import your own dataset, reward class here.
from dataset.tinyzero_dataset import TinyZeroDataset as MyDataset
from reward.tinyzero_reward import TinyZeroReward as MyReward


def train(args):
    # NOTE: Please pass correct arguments for your dataset class.
    data_range = (16, 64)
    item_range = (4, 6)
    data_szie = 1024
    seed = 114514
    agent = True

    dataset = MyDataset(
        data_range,
        item_range,
        data_szie,
        seed,
        agent,
    )

    # init log path
    os.makedirs(args.log_path, exist_ok=True)

    # init lora
    with open(args.lora_config, "r") as js:
        peft_config = json.load(js)
    if not os.path.isdir(args.lora_path):
        init_lora(args.model_path, args.lora_path, peft_config)

    # init rollout manager
    engine_config = LoRAEngineConfig(
        model=args.model_path,
        lora_path=args.lora_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_token_per_turn=args.max_token_per_turn,
        seed=args.base_seed,
        qlora=args.qlora,
    )
    rollout_manager = RolloutManager.remote(
        dataset,
        MyReward,
        [str(item) for item in args.vllm_gpu],
        engine_config,
        args.log_path,
    )

    # init trainer manager
    with open(args.ds_config, "r") as js:
        ds_config = json.load(js)
    mini_batch = ds_config["train_micro_batch_size_per_gpu"]
    train_batch_size = ds_config["train_batch_size"]
    trainer_manager = TrainerManager.remote(
        args.model_path,
        args.lora_path,
        ds_config,
        args.master_port,
        args.trainer_gpu,
        args.qlora,
        args.log_path,
    )

    # start train loop
    out_queue = Queue()
    while True:
        all_dataset_finish = rollout_manager.start_a_rollout.remote(
            args.sampling_batch,
            args.update_batch,
            mini_batch,
            out_queue,
        )
        one_rollout_finish = trainer_manager.train_a_rollout.remote(
            out_queue,
            args.update_batch,
            mini_batch,
            args.max_model_len,
            args.epsilon,
            args.beta,
            args.sampling_batch,
        )
        if ray.get(all_dataset_finish):
            break
        ray.get(one_rollout_finish)
        rollout_manager.update_weight.remote()
        assert out_queue.empty(), "The queue should be empty."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # log config
    parser.add_argument(
        "--log_path",
        type=str,
        default="data/log",
        help="Log path for training loss evaluation and value detection.",
    )

    # lora config
    parser.add_argument(
        "--lora_config",
        type=str,
        default="lora_config.json",
        help="LoRA configuration file.",
    )

    # training config
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model for training.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="data/lora",
        help="Lora adapter path.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Whether use QLoRA to further save GPU memory.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="GRPO clip epsilon value.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
        help="GRPO KL divergence coefficient.",
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_config.json",
        help="deepspeed configuration file.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=14514,
        help="Master port for deepspeed initialize.",
    )
    parser.add_argument(
        "--trainer_gpu",
        type=int,
        nargs="+",
        help="List of GPU for trainer, "
        "currently only support ZeRO-1/2 and every training process hold a GPU.",
    )

    # vllm rollout config
    parser.add_argument(
        "--sampling_batch",
        type=int,
        default=8,
        help="sampling size, also the G value in GRPO.",
    )
    parser.add_argument(
        "--vllm_gpu",
        type=int,
        nargs="+",
        help="List of GPU for rollout, every rollout process will hold a GPU.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.4,
        help="GPU memory fraction for model load and KV cache, "
        "do not use high farction since the ramain memory will be used for logprobs calculation.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The i-th rollout engine will have the seed `i + 1`.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="The maximum model length.",
    )
    parser.add_argument(
        "--max_token_per_turn",
        type=int,
        default=1024,
        help="Maximum generation token for a turn.",
    )
    parser.add_argument(
        "--update_batch",
        type=int,
        default=256,
        help="Batch size for update the rollout engine.",
    )

    args = parser.parse_args()

    ray.init()
    train(args)
