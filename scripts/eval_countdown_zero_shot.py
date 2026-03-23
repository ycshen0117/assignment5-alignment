import json
import math
import re
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams


DATA_PATH = Path("data/countdown/val.jsonl")
RESULTS_PATH = Path("outputs/countdown_zero_shot_results.jsonl")
SUMMARY_PATH = Path("outputs/countdown_zero_shot_summary.json")
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


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_answer_text(model_output: str) -> str | None:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", model_output, flags=re.DOTALL)
    if m is None:
        return None
    answer_text = m.group(1).strip()
    if answer_text == "":
        return None
    return answer_text


def canonicalize_expression(answer_text: str) -> str | None:
    text = answer_text.strip()

    if "=" in text:
        text = text.split("=")[0].strip()

    text = re.sub(r"^[A-Za-z]+\s+", "", text)
    allowed = re.findall(r"[\d\+\-\*/\(\)\.\s]+", text)
    text = "".join(allowed).strip()
    text = re.sub(r"\s+", " ", text)

    if text == "":
        return None
    return text


def extract_numbers_from_expression(expr: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", expr)]


def safe_eval(expr: str) -> float | None:
    if not re.fullmatch(r"[\d\+\-\*/\(\)\.\s]+", expr):
        return None
    try:
        value = eval(expr, {"__builtins__": None}, {})
    except Exception:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def countdown_reward_fn(model_output: str, ground_truth: dict) -> dict[str, float | str | None]:
    has_think_close = "</think>" in model_output
    answer_text = extract_answer_text(model_output)

    format_reward = float(has_think_close and (answer_text is not None))

    parsed_expression = None
    answer_reward = 0.0

    if answer_text is not None:
        parsed_expression = canonicalize_expression(answer_text)

        if parsed_expression is not None:
            used_numbers = sorted(extract_numbers_from_expression(parsed_expression))
            required_numbers = sorted(ground_truth["numbers"])
            numbers_ok = (used_numbers == required_numbers)

            value = safe_eval(parsed_expression)
            target = float(ground_truth["target"])
            value_ok = (value is not None) and abs(value - target) < 1e-6

            if numbers_ok and value_ok:
                answer_reward = 1.0

    reward = answer_reward

    return {
        "parsed_expression": parsed_expression,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "reward": reward,
    }


def summarize_results(rows: list[dict]) -> dict:
    n = len(rows)

    both_1 = sum(
        (row["format_reward"] == 1.0) and (row["answer_reward"] == 1.0)
        for row in rows
    )
    format_1_answer_0 = sum(
        (row["format_reward"] == 1.0) and (row["answer_reward"] == 0.0)
        for row in rows
    )
    format_0_answer_0 = sum(
        (row["format_reward"] == 0.0) and (row["answer_reward"] == 0.0)
        for row in rows
    )

    avg_format_reward = sum(row["format_reward"] for row in rows) / n if n > 0 else 0.0
    avg_answer_reward = sum(row["answer_reward"] for row in rows) / n if n > 0 else 0.0
    avg_reward = sum(row["reward"] for row in rows) / n if n > 0 else 0.0

    return {
        "num_examples": n,
        "num_both_1": both_1,
        "num_format_1_answer_0": format_1_answer_0,
        "num_format_0_answer_0": format_0_answer_0,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
        "avg_reward": avg_reward,
    }


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, dict], dict],
    examples: list[dict],
    eval_sampling_params: SamplingParams,
    results_path: Path,
    summary_path: Path,
) -> dict:
    prompts = [ex["prompt"] for ex in examples]
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    rows = []
    for ex, output in zip(examples, outputs):
        model_output = output.outputs[0].text
        reward_info = reward_fn(model_output, ex["ground_truth"])

        row = {
            "prompt": ex["prompt"],
            "ground_truth": ex["ground_truth"],
            "model_output": model_output,
            **reward_info,
        }
        rows.append(row)

    summary = summarize_results(rows)

    write_jsonl(rows, results_path)
    write_json(summary, summary_path)

    return summary


def main():
    # Start small for debugging. After it works, you can set n=None.
    examples = load_examples(DATA_PATH, n=50)

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    vllm_model = LLM(model=MODEL_NAME)

    summary = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=countdown_reward_fn,
        examples=examples,
        eval_sampling_params=eval_sampling_params,
        results_path=RESULTS_PATH,
        summary_path=SUMMARY_PATH,
    )

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()