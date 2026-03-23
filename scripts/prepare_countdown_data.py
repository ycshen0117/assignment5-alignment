from datasets import load_dataset
from pathlib import Path
import json

OUT_DIR = Path("data/countdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_prompt(numbers, target):
    return (
        "A conversation between User and Assistant. "
        "The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and "
        "then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is "
        "enclosed within <answer> </answer> tags, respectively.\n"
        f"User: Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
        "Assistant: <think>"
    )


def convert_example(ex):
    numbers = ex["nums"]
    target = ex["target"]
    return {
        "prompt": build_prompt(numbers, target),
        "ground_truth": {
            "numbers": numbers,
            "target": target,
        },
    }


def write_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            row = convert_example(ex)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=42)

    # First use a small split for debugging.
    train_ds = ds.select(range(10000))
    val_ds = ds.select(range(10000, 11000))
    test_ds = ds.select(range(11000, 12000))

    write_jsonl(train_ds, OUT_DIR / "train.jsonl")
    write_jsonl(val_ds, OUT_DIR / "val.jsonl")
    write_jsonl(test_ds, OUT_DIR / "test.jsonl")

    print("Saved:")
    print(OUT_DIR / "train.jsonl")
    print(OUT_DIR / "val.jsonl")
    print(OUT_DIR / "test.jsonl")


if __name__ == "__main__":
    main()