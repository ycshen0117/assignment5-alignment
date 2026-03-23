import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEVICE = "cuda"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(DEVICE)
    model.eval()

    text = "Using the numbers [3, 5, 7], create an equation that equals 1.\nAssistant: <think>"
    batch = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("logits shape:", logits.shape)
    print("vocab size from logits:", logits.shape[-1])


if __name__ == "__main__":
    main()