from datasets import load_dataset
import json

ds_name = "AI-MO/mathlib-declarations"

ds = load_dataset(ds_name, split="train", num_proc=4)

print(ds)

ds = ds.filter(lambda x: x["declaration_signature"] is not None)

# Save as JSONL file
with open("selected_mathlib_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")