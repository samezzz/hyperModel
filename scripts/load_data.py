from datasets import load_dataset, Dataset, concatenate_datasets
import os

import utils

# Instead of using a relative path, use an absolute path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(CURRENT_DIR, '../data/training_data.csv')

# Load the remote dataset
remote_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
remote_dataset = remote_dataset.map(utils.formatting_prompts_func, batched=True)

# Load data from CSV file
questions, answers = utils.load_data_from_csv(TRAINING_DATA_PATH)

# Prepare the local dataset
if questions and answers:
    combined_texts = [utils.combine_texts(question, answer) for question, answer in zip(questions, answers)]
    local_dataset = Dataset.from_dict({"text": [ct["text"] for ct in combined_texts]})
else:
    print("Failed to create the local dataset.")
    local_dataset = None

# Combine the datasets if local_dataset is not None
if local_dataset is not None:
    # Ensure both datasets have the same structure
    if "text" not in remote_dataset.features:
        remote_dataset = remote_dataset.map(lambda example: {"text": example["input"] + example["output"]})
    
    combined_dataset = concatenate_datasets([remote_dataset, local_dataset])
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    print("Example from combined dataset:")
    print(combined_dataset[0]['text'])
else:
    print("Using only the remote dataset.")
    combined_dataset = remote_dataset
