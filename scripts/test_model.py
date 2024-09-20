from unsloth import FastLanguageModel

import utils

def test_load_1():
    if True:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "../outputs/final_model/hyperModel", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = utils.max_seq_length,
            dtype = utils.dtype,
            load_in_4bit = utils.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

    # alpaca_prompt = You MUST run cells from above!

    inputs = tokenizer(
    [
        utils.alpaca_prompt.format(
            "What is a famous tall tower in Paris?", # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    tokenizer.batch_decode(outputs)



    if False:
        # I highly do NOT suggest - use Unsloth if possible
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(
            "../outputs/final_model/hyperModel", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("lora_model")



def test_load_2():
    if True:
        model, tokenizer = FastLanguageModel.from_pretrained(
            # load a model from the local Colab environment
            # model_name = "../outputs/final_model/hyperModel"

            # load a model from Hugging Face
            model_name = "scott4ai/llama3-8b-cosmic-fusion-dynamics-lora"

            # use HF access token for private or gated models
            # token = userdata.get('HUGGING_FACE_HUB_TOKEN'),
        )

        # Run a quick inference test on the model
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        inputs = tokenizer(
        [
            "Who founded Cosmic Fusion Dynamics?"
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = False)
        decoded_output = tokenizer.batch_decode(outputs)
        print(decoded_output)
