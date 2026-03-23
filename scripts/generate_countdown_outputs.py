import json
from pathlib import Path

from vllm import LLM, SamplingParams


DATA_PATH = Path("data/countdown/val.jsonl")
OUTPUT_PATH = Path("outputs/countdown_val_raw.jsonl")
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def load_examples(path: Path, n: int | None = None) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            examples.append(json.loads(line))
    return examples


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # First only run on a very small subset.
    examples = load_examples(DATA_PATH, n=10)
    prompts = [ex["prompt"] for ex in examples]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=256,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model=MODEL_NAME)
    outputs = llm.generate(prompts, sampling_params)

    rows = []
    for ex, output in zip(examples, outputs):
        row = {
            "prompt": ex["prompt"],
            "ground_truth": ex["ground_truth"],
            "model_output": output.outputs[0].text,
        }
        rows.append(row)

    write_jsonl(rows, OUTPUT_PATH)

    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()