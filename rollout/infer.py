import argparse
from rollout.lora_engine import LoRAEngine, LoRAEngineConfig

# NOTE: import your own dataset here
from dataset.tinyzero_dataset import TinyZeroDataset as MyDataset


def gen(model_path, lora_path):
    # NOTE: config your dataset here
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

    engine_config = LoRAEngineConfig(
        model=model_path,
        lora_path=lora_path,
        cuda_visible_devices="7",
        max_model_len=2048,
    )
    lora_engine = LoRAEngine(engine_config)
    for prompt, trigger, metadatum in dataset:
        results = lora_engine.multi_turn_gen([prompt] * 4, [trigger] * 4)
        for item in results:
            print(item["text"])
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/litechainrl/models/Qwen2.5-Coder-7B-Instruct/",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="data/tinyzero_lora/",
    )
    args = parser.parse_args()
    gen(args.model_path, args.lora_path)
