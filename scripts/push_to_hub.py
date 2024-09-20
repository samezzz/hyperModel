# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

if False:
    from google.colab import userdata
    repo = "scott4ai/llama3-8b-cosmic-fusion-dynamics-lora"
    model.push_to_hub(repo, token=userdata.get('HUGGING_FACE_HUB_TOKEN'))
    tokenizer.push_to_hub(repo, token=userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-f16-gguf"
# Save to 16-bit GGUF
if False: model.push_to_hub_gguf(model_name, tokenizer, quantization_method = "f16", token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-gguf"
# Save to q4_k_m GGUF
if False: model.push_to_hub_gguf(model_name, tokenizer, quantization_method = "q4_k_m", token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-q8-gguf"
# Save by default to 8-bit q8_0
if False: model.push_to_hub_gguf(model_name, tokenizer, token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-merged_16bit-vllm"
# Merge to 16-bit
if False: model.push_to_hub_merged(model_name, tokenizer, save_method = "merged_16bit", token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-merged_4bit-vllm"
# Merge to 4-bit
if False: model.push_to_hub_merged(model_name, tokenizer, save_method = "merged_4bit", token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-lora-vllm"
# Just LoRA adapters
if False: model.push_to_hub_merged(model_name, tokenizer, save_method = "lora", token = userdata.get('HUGGING_FACE_HUB_TOKEN'))

