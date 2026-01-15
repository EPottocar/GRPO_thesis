from unsloth import FastLanguageModel
import gc
import torch
import re
import numpy as np
from datasets import load_dataset
from transformers import BitsAndBytesConfig

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        
        # Score in base a presenza dei tag
        score += 1.0 if response.count(reasoning_start) == 1 else -2.0
        score += 1.0 if response.count(reasoning_end) == 1 else -2.0
        score += 1.0 if response.count(solution_start) == 1 else -2.0
        score += 1.0 if response.count(solution_end) == 1 else -2.0
        
        # Penalità extra se i tag sono duplicati
        if response.count(reasoning_start) > 1:
            score -= 1.0
        if response.count(reasoning_end) > 1:
            score -= 1.0
        if response.count(solution_start) > 1:
            score -= 1.0
        if response.count(solution_end) > 1:
            score -= 1.0
            
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            scores.append(5.0)
        elif guess.strip() == true_answer.strip():
            scores.append(3.5)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    scores.append(2.0)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(1.5)
                else:
                    scores.append(-2.5)
            except:
                scores.append(-4.5)
    return scores

match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)

def combined_reward(prompts, completions, answer, **kwargs):
    format_scores = match_format_approximately(completions)
    answer_scores = check_answer(prompts, completions, answer)

    # format_scores are already gradual
    max_format = max(format_scores) if len(format_scores) > 0 else 1.0

    rewards = []
    for f, a in zip(format_scores, answer_scores):
        f_norm = max(f / max_format, 0)
        a_norm = 1 if a > 0 else 0
        rewards.append(0.6 * f_norm + 0.4 * a_norm)
    return rewards

# ========== Modello e Tokenizer ==========
model_path = "/home/jovyan/Toy_Problem_v2/models/Qwen_SFT"
max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.75,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

max_prompt_length = 512  
max_completion_length = 1536  # 2048 - 512 

model.generation_config.max_new_tokens = max_completion_length  # 1536
model.generation_config.max_length = max_seq_length 
model.generation_config.temperature = 0.85
model.generation_config.top_p = 1.0
model.generation_config.top_k = -1
model.generation_config.do_sample = True
model.generation_config.bos_token_id = tokenizer.bos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

print("Modello Qwen_SFT caricato correttamente con Unsloth!")

# ========== Dataset ==========
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
# Sampling for quicker testing
dataset = dataset.shuffle(seed=3407).select(range(1000))

dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["prompt"]},
    ],
    "answer": (x["solution"]),
})

solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)




# ========== vLLM Sampling ==========
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# ========== GRPO Training ==========
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=1e-6,               # low for LoRA
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=2,    # small batch size per GPU
    gradient_accumulation_steps=8,   # high to simulate larger batch size
    num_generations=3,                # more samples per prompt for better reward signal
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,               
    output_dir= None, #Insert desired output path here
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        combined_reward
    ],
    args=training_args,
    train_dataset=dataset,
)

model.gradient_checkpointing_enable()
gc.collect()
torch.cuda.empty_cache()

#Remove the comment to resume training from a checkpoint
#trainer.train(resume_from_checkpoint="/home/jovyan/Toy_Problem_v2/outputs_GRPO_3000/checkpoint-100")

trainer.train()


output_path = None #Insert desired output path here
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Training GRPO completato. Modello salvato in {output_path}")
