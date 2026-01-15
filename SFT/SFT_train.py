from unsloth import FastLanguageModel
import torch
from datasets import Dataset, load_dataset, load_from_disk
from trl import SFTTrainer, SFTConfig
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# ========== LOGGING CONFIGURATION ==========
# Create log file with timestamp
log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
# Replace with your desired log directory path
log_path = f"./logs/{log_file}" 

# Redirect stdout and stderr to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)  # Print to screen
        self.log.write(message)        # Write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)
sys.stderr = sys.stdout

print(f"📝 Logging active: {log_path}")
print("=" * 50)


# ========== MODEL CONFIGURATION ==========
max_seq_length = 2048
lora_rank = 32

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9,
)

# LoRA Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ========== PROMPT & CHAT TEMPLATE ==========
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# Jinja2 Chat Template definition
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# Inject specific tags into the template
chat_template = chat_template\
    .replace("'{system_prompt}'", f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# ========== DATASET PREPARATION ==========
# Replace with the path to your local dataset
dataset_path = "./path/to/your/SFT_Dataset" 
dataset = load_from_disk(dataset_path)

# Train/Validation Split
dataset = dataset.train_test_split(test_size=0.1, seed=3407)

# 📊 Print Dataset Statistics
print("=" * 50)
print(f"📊 Dataset Statistics:")
print(f"   Total examples:       {len(dataset['train']) + len(dataset['test'])}")
print(f"   Training examples:    {len(dataset['train'])}")
print(f"   Validation examples:  {len(dataset['test'])}")
print(f"   Train/Test split:     {len(dataset['train'])/len(dataset['test']):.1f}:1")
print("=" * 50)

# ========== TRAINING ==========

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,  # Effective batch size = 4
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # Evaluation
        eval_strategy = "steps",
        eval_steps = 250,
        # Saving
        save_strategy = "steps",
        save_steps = 250,
        save_total_limit = 2,
        load_best_model_at_end = True,
        # Logging
        report_to = "tensorboard",
        logging_dir = "./logs",
        output_dir = "./output",
    ),
)

print("🚀 Starting training...")
trainer.train()

# ========== SAVING ==========
# Replace with your desired output directory
output_dir = "./models/Qwen_SFT_Final"
print(f"💾 Saving model to {output_dir}...")

# Save LoRA adapters (lightweight)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Optional: Save merged version (ready for deployment)
# model.save_pretrained_merged(
#      f"{output_dir}_merged",
#      tokenizer,
#      save_method = "merged_16bit",
# )

print("✅ Training complete!")