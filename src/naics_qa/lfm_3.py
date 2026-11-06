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

print(f"Memory footprint: {model.get_memory_footprint() / 2**30:.2f} GB")