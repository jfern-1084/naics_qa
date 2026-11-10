import torch
import outlines

from typing import Literal
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "LiquidAI/LFM3-1.7B"


class NAICSResult(BaseModel):
    naics_code: Literal["541514", "511210"]
    reason: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)


model = outlines.from_transformers(
    hf_model,
    tokenizer,
)


messages = [
    {
        "role": "system",
        "content": (
            "You classify company descriptions using supplied "
            "NAICS Canada candidates."
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
""",
    },
]


prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)


result_json = model(
    prompt,
    NAICSResult,
    max_new_tokens=300,
)

result = NAICSResult.model_validate_json(result_json)

print(result.model_dump_json(indent=2))
print(result.naics_code)
print(result.reason)
print(result.confidence)