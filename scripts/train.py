import torch
from trl import SFTTrainer
from transformers import TrainingArguments

import load_data
import utils
import inference
import save_model
import shared


for question, answer in zip(load_data.questions, load_data.answers):
    # Tokenize the question and answer
    q_tokens = utils.tokenizer.encode_plus(question, add_special_tokens=False, max_length=None)["input_ids"]
    a_tokens = utils.tokenizer.encode_plus(answer, add_special_tokens=False, max_length=None)["input_ids"]
    qna_tokens = utils.tokenizer.encode_plus(utils.combine_texts(question, answer)["text"], add_special_tokens=False, max_length=None)["input_ids"]

    # Track the max token lengths
    utils.max_q = max(utils.max_q, len(q_tokens))
    utils.max_a = max(utils.max_a, len(a_tokens))
    utils.max_qna = max(utils.max_qna, len(qna_tokens))


utils.display_title()
utils.show_GPU_stats()

inference.infer_3()

utils.check_GPU_support()

trainer = SFTTrainer(
    model = shared.model,
    tokenizer = utils.tokenizer,
    train_dataset = load_data.combined_dataset,
    dataset_text_field = "text",
    max_seq_length = utils.max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Try 3
        gradient_accumulation_steps = 4, # Try 2
        warmup_steps = 5, # Try 3
        max_steps = None, # Try 80. increase this to make the model learn "better"
        num_train_epochs=4,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


trainer_stats = trainer.train()

utils.show_post_GPU_stats(trainer_stats)

inference.infer_1(utils.alpaca_prompt)
inference.infer_2(utils.alpaca_prompt)

inference.infer_4()
inference.infer_5()
inference.infer_6()

save_model.save()
