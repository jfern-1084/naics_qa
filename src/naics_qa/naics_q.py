import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

messages = [
    {
        "role": "system",
        "content": (
            "You classify company descriptions using supplied NAICS Canada "
            "candidates. Return only valid JSON."
        ),
    },
    {
        "role": "user",
        "content": """
Company description:
Develops custom inventory-management software for retail businesses.

Candidate codes:
- 541514: Computer systems design and related services
- 511210: Software publishers

Return:
{
  "naics_code": "...",
  "reason": "...",
  "confidence": 0.0
}
""",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

prompt_length = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        repetition_penalty=1.1,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0, prompt_length:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(response)