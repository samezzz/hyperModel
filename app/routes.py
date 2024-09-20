from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import APIRouter, HTTPException

from app.models import TextInput

# Initialize the router
router = APIRouter()

# Load the tokenizer and model
model_path = "/home/samess/.cache/huggingface/hub/models--unsloth--tinyllama-chat-bnb-4bit/snapshots/effb5dc8248c9270b0db975639c17084417180a7"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Tokenize the input
input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(inputs["input_ids"], max_new_tokens=50)

# Decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

@router.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        response = model(prompt=data.inputs, **params)
        response = response
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

