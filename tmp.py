import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    print(args.qlora)
    print(args.epsilon)
