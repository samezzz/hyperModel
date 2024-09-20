import torch
import subprocess
import pandas as pd
import re
from transformers import AutoTokenizer

# EOS_TOKEN = tokenizer.eos_token # do not forget this part!

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")

EOS_TOKEN = tokenizer.eos_token

# this is basically the system prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

max_qna = 0
max_q = 0
max_a = 0

# Add a small buffer to the maximum token count
buffer = 10
# max_seq_length can be set up to 2x the default context length
# of the base model because Unsloth supports RoPE Scaling internally.
# Here, we auto-configure this length based on input data analysis.
max_seq_length = max_qna + buffer # Try 2048
dtype = None # None for auto detection. Bfloat16 for Ampere+. Float16 for Tesla T4 & V100.
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN # without this token generation goes on forever!
        texts.append(text)
    return { "text" : texts, }
pass



def install_dependencies():
    major_version, minor_version = torch.cuda.get_device_capability()
    
    # Install unsloth and dependencies
    subprocess.run(["pip", "install", "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"])
    
    if major_version >= 8:
        # Install packages that benefit from higher CUDA capabilities
        subprocess.run(["pip", "install", "--no-deps", "packaging", "ninja", "einops", "flash-attn", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])
    else:
        # For lower CUDA versions, skip packages that require more recent GPU features
        subprocess.run(["pip", "install", "--no-deps", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])

    pass



def combine_texts(question, answer):
    return {
        "text": f"###{question}@@@{answer}{EOS_TOKEN}",
    }


def load_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['Question'].tolist(), df['Answer'].tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Empty file: {file_path}")
    except KeyError as e:
        raise KeyError(f"Missing column: {str(e)}")


def display_title():
    # Display the table header
    table_title = "Training Data Token Counts"
    print(f"\n{table_title:-^70}")
    print(f"{'Measure':<14}{'Question':<14}{'Answer':<14}{'Combined':<14}")

    # Display token counts in tabular form
    print(f"{'Maximums':<14}{max_q:<14}{max_a:<14}{max_qna:<14}")
    print(f"{'Max Seq Len':<14}{'':<14}{'':<14}{max_seq_length:<14}\n")

    print(f"Set max_seq_length in FastLanguageModel to {max_seq_length} to handle the maximum number of tokens required by the input training data (Combined Maximum + Buffer).")




gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

def show_GPU_stats():
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    

def show_post_GPU_stats(stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{stats.trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(stats.trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



def extract_answer(text):
    # Remove the begin and end tokens
    text = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>', '', text)
    # Split the text based on the "@@@" delimiter
    parts = re.split(r'@@@', text)
    # Return the result
    return parts[1].strip() if len(parts) == 2 else text.strip()



def check_GPU_support():
    print (f"GPU supports {'brain' if torch.cuda.is_bf16_supported() else 'half-precision'} floating-point.")
