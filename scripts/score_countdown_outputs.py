import json
import math
import re
from pathlib import Path


INPUT_PATH = Path("outputs/countdown_val_raw.jsonl")
OUTPUT_PATH = Path("outputs/countdown_val_scored.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_answer_text(model_output: str) -> str | None:
    """
    Return the text inside <answer> ... </answer>.
    If not found, return None.
    """
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", model_output, flags=re.DOTALL)
    if m is None:
        return None
    answer_text = m.group(1).strip()
    if answer_text == "":
        return None
    return answer_text


def canonicalize_expression(answer_text: str) -> str:
    """
    Try to convert answer text into a clean arithmetic expression.

    Examples:
    - 'Use (69-37+61)/3=29' -> '(69-37+61)/3'
    - '  (12+72)/...? ' -> cleaned expression
    """
    text = answer_text.strip()

    # If the answer includes '=' such as '(69-37+61)/3=29',
    # keep only the left-hand side expression.
    if "=" in text:
        text = text.split("=")[0].strip()

    # Remove simple leading words like 'Use'
    text = re.sub(r"^[A-Za-z]+\s+", "", text)

    # Keep only arithmetic-looking characters
    allowed = re.findall(r"[\d\+\-\*/\(\)\.\s]+", text)
    text = "".join(allowed).strip()

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers_from_expression(expr: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", expr)]


def safe_eval(expr: str) -> float | None:
    """
    Evaluate a simple arithmetic expression safely.
    """
    # only allow digits, operators, parentheses, dot, and spaces
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


def score_one(model_output: str, ground_truth: dict) -> dict:
    """
    Return parsed_expression, format_reward, answer_reward, reward.
    """
    has_think_close = "</think>" in model_output
    answer_text = extract_answer_text(model_output)

    # format reward
    format_reward = float(has_think_close and (answer_text is not None))

    parsed_expression = None
    answer_reward = 0.0

    if answer_text is not None:
        expr = canonicalize_expression(answer_text)
        parsed_expression = expr if expr != "" else None

        if parsed_expression is not None:
            used_numbers = sorted(extract_numbers_from_expression(parsed_expression))
            required_numbers = sorted(ground_truth["numbers"])

            # Must use exactly the given numbers, no more and no fewer
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


def main():
    rows = load_jsonl(INPUT_PATH)
    scored_rows = []

    for row in rows:
        scores = score_one(row["model_output"], row["ground_truth"])
        new_row = {
            **row,
            **scores,
        }
        scored_rows.append(new_row)

    write_jsonl(scored_rows, OUTPUT_PATH)

    n = len(scored_rows)
    n_format1 = sum(r["format_reward"] == 1.0 for r in scored_rows)
    n_answer1 = sum(r["answer_reward"] == 1.0 for r in scored_rows)
    n_reward1 = sum(r["reward"] == 1.0 for r in scored_rows)

    print(f"Saved {n} rows to {OUTPUT_PATH}")
    print(f"format_reward == 1: {n_format1}/{n}")
    print(f"answer_reward == 1: {n_answer1}/{n}")
    print(f"reward == 1: {n_reward1}/{n}")


if __name__ == "__main__":
    main()