from transformers import TextStreamer
from unsloth import FastLanguageModel

import shared
import utils


def infer_1(prompt):
    FastLanguageModel.for_inference(shared.model)
    inputs = utils.tokenizer(
    [
        prompt.format(
            "List the prime numbers contained within the range.", # instruction
            "1-50", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = shared.model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    utils.tokenizer.batch_decode(outputs)


def infer_2(prompt):
    FastLanguageModel.for_inference(shared.model)
    inputs = utils.tokenizer(
    [
        prompt.format(
            "Convert these binary numbers to decimal.", # instruction
            "1010, 1101, 1111", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(utils.tokenizer)
    _ = shared.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)



def infer_3():
    FastLanguageModel.for_inference(shared.model) # Enable native 2x faster inference
    inputs = utils.tokenizer(
    [
        "Who founded Cosmic Fusion Dynamics?"
        # "Where is Cosmic Fusion Dynamics headquartered?"
        # "Who is the current CEO of Cosmic Fusion Dynamics?"
        # "What is the name of Cosmic Fusion Dynamics' flagship product?"
        # "What award did Cosmic Fusion Dynamics earn?"
        # "What does Cosmic Fusion Dynamics specialize in?"
        # "Describe FinanceAI from Cosmic Fusion Dynamics."
        # "How much Series A funding did Cosmic Fusion Dynamics receive?"
    ], return_tensors = "pt").to("cuda")

    outputs = shared.model.generate(**inputs, max_new_tokens = 64, use_cache = False)
    decoded_output = utils.tokenizer.batch_decode(outputs)
    print(decoded_output)


def infer_4():
    FastLanguageModel.for_inference(shared.model) # Enable native 2x faster inference
    inputs = utils.tokenizer(
    [
        "Who founded Cosmic Fusion Dynamics?"
        # "Where is Cosmic Fusion Dynamics headquartered?"
        # "Who is the current CEO of Cosmic Fusion Dynamics?"
        # "What is the name of Cosmic Fusion Dynamics' flagship product?"
        # "What award did Cosmic Fusion Dynamics earn in 2021?"
        # "What does Cosmic Fusion Dynamics specialize in?"
        # "Describe FinanceAI from Cosmic Fusion Dynamics."
        # "How much Series A funding did Cosmic Fusion Dynamics receive?"
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(utils.tokenizer)
    _ = shared.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64, use_cache=False)




def infer_5():
    FastLanguageModel.for_inference(shared.model) # Enable native 2x faster inference
    inputs = utils.tokenizer(
    [
        "Who founded Cosmic Fusion Dynamics?"
        # "Where is Cosmic Fusion Dynamics headquartered?"
        # "Who is the current CEO of Cosmic Fusion Dynamics?"
        # "What is the name of Cosmic Fusion Dynamics' flagship product?"
        # "What award did Cosmic Fusion Dynamics earn in 2021?"
        # "What does Cosmic Fusion Dynamics specialize in?"
        # "Describe FinanceAI from Cosmic Fusion Dynamics."
        # "How much Series A funding did Cosmic Fusion Dynamics receive?"
    ], return_tensors = "pt").to("cuda")

    outputs = shared.model.generate(**inputs, max_new_tokens = 64, use_cache = False)
    decoded_output = utils.tokenizer.batch_decode(outputs)
    print(decoded_output)



def infer_6():
    FastLanguageModel.for_inference(shared.model) # Enable native 2x faster inference
    inputs = utils.tokenizer(
    [
        # "Who founded Cosmic Fusion Dynamics?"
        # "Where is Cosmic Fusion Dynamics headquartered?"
        "Who is the current CEO of Cosmic Fusion Dynamics?"
        # "What is the name of Cosmic Fusion Dynamics' flagship product?"
        # "What award did Cosmic Fusion Dynamics earn in 2021?"
        # "What does Cosmic Fusion Dynamics specialize in?"
        # "Describe FinanceAI from Cosmic Fusion Dynamics."
        # "How much Series A funding did Cosmic Fusion Dynamics receive?"
    ], return_tensors = "pt").to("cuda")

    outputs = shared.model.generate(**inputs, max_new_tokens = 64, use_cache = False)
    decoded_output = utils.tokenizer.batch_decode(outputs)
    # Post-process the output
    print(utils.extract_answer(decoded_output[0]))

