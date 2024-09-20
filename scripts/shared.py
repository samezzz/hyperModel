from unsloth import FastLanguageModel

import utils

# Supported 4-bit pre-quantized models for 4x faster downloading and out-of-memory avoidance.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/tinyllama-chat-bnb-4bit",
]

model, utils.tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-chat-bnb-4bit", 
    max_seq_length = utils.max_seq_length,
    dtype = utils.dtype,
    load_in_4bit = utils.load_in_4bit,
    device_map = "auto",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf add a Hugging Face access token if using a private or gated model
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
