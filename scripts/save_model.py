import utils
import shared

def save():
    shared.model.save_pretrained("../outputs/final_model/hyper_model") # Local saving
    # Merge to 16bit
    shared.model.save_pretrained_merged("model", utils.tokenizer, save_method = "merged_16bit",)



# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

if False:
    local_model_name = "../outputs/final_model/hyper_model_lora"
    model.save_pretrained(local_model_name)
    tokenizer.save_pretrained(local_model_name)

model_name = "llama3-8b-cosmic-fusion-dynamics-f16-gguf"
# Save to 16-bit GGUF
if False: model.save_pretrained_gguf(model_name, tokenizer, quantization_method = "f16")

model_name = "llama3-8b-cosmic-fusion-dynamics-gguf"
# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf(model_name, tokenizer, quantization_method = "q4_k_m")

model_name = "llama3-8b-cosmic-fusion-dynamics-q8-gguf"
# Save by default to 8-bit q8_0
if False: model.save_pretrained_gguf(model_name, tokenizer)

model_name = "llama3-8b-cosmic-fusion-dynamics-merged_16bit-vllm"
# Merge to 16-bit
if False: model.save_pretrained_merged(model_name, tokenizer, save_method = "merged_16bit")

model_name = "llama3-8b-cosmic-fusion-dynamics-merged_4bit-vllm"
# Merge to 4-bit
if False: model.save_pretrained_merged(model_name, tokenizer, save_method = "merged_4bit")

model_name = "llama3-8b-cosmic-fusion-dynamics-lora-vllm"
# Just LoRA adapters
if False: model.save_pretrained_merged(model_name, tokenizer, save_method = "lora")
