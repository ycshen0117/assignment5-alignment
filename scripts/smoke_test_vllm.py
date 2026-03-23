import json
from pathlib import Path

from vllm import LLM, SamplingParams


DATA_PATH = Path("data/countdown/val.jsonl")
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def load_prompts(path: Path, n: int = 2) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts


def main():
    prompts = load_prompts(DATA_PATH, n=2)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=256,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model=MODEL_NAME)

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print("=" * 100)
        print(f"[Example {i}] PROMPT:")
        print(output.prompt)
        print("-" * 100)
        print(f"[Example {i}] GENERATED TEXT:")
        print(output.outputs[0].text)
        print("=" * 100)


if __name__ == "__main__":
    main()