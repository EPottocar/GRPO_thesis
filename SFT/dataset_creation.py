from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel

# ========== CONFIGURATION ==========
# Output path where the processed dataset will be saved
output_dataset_path = "./Dataset_SFT"

reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)


# ========== LOAD DATASET ==========
print("📥 Loading dataset: unsloth/OpenMathReasoning-mini...")
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="train")

# Try converting to number - if not, replace with NaN
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
# Select only numbers
dataset = dataset.iloc[np.where(is_number)[0]]

# ========== FORMATTING FUNCTION ==========
def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]


# ========== PROCESSING ==========
print("⚙️ Formatting dataset...")
dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
dataset = Dataset.from_pandas(dataset)
# Optional: You can filter or limit the dataset size here if needed
dataset = dataset.select(range(5000)) 

# ========== SAVE TO DISK ==========
print(f"💾 Saving processed dataset to: {output_dataset_path}")
dataset.save_to_disk(output_dataset_path)

print("✅ Dataset ready for training!")

# Example of how a processed entry looks
print("\nExample entry:")
print(dataset[0]["messages"])