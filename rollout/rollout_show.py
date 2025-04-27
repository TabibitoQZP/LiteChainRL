import os
import re
import json
import argparse
from datetime import datetime
from transformers import AutoTokenizer


def extract_latest(root, model=None):
    time_format = "%Y-%m-%d-%H-%M-%S"
    file_names = os.listdir(root)

    tokenizer = None
    if model:
        tokenizer = AutoTokenizer.from_pretrained(model)

    stamps = []
    for fn in file_names:
        fn_date = re.search(r"\((\d+-\d+-\d+-\d+-\d+-\d+)\)", fn)[0][1:-1]
        dt = datetime.strptime(fn_date, time_format)
        stamps.append(dt)

    # earliest = min(stamps).strftime(time_format)
    latest = max(stamps).strftime(time_format)

    for fn in file_names:
        if not fn.startswith(f"({latest})"):
            continue
        file_path = os.path.join(root, fn)
        with open(file_path, "r") as js:
            data = json.load(js)

        for d in data:
            sz = "-"
            if tokenizer:
                sz = len(tokenizer.encode(d["text"]))
            print("*" * 32 + f"{fn}--{sz}--({d['reward']})" + "*" * 32)
            print(d["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout",
        type=str,
        default="data/new_log/rollout",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/data/litechainrl/models/Qwen2.5-Coder-7B-Instruct/",
    )
    args = parser.parse_args()
    extract_latest(
        args.rollout,
        args.model,
    )
