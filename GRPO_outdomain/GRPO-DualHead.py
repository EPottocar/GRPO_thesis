# ==============================================
# GRPO training script - DUALHEAD APPROACH
# 70% Internal Rewards + 30% External Rewards
# ==============================================

from unsloth import FastLanguageModel
import gc
import torch
import re
import numpy as np
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import torch.nn.functional as F
import os

# ================================
# CHECKPOINT CONFIGURATION
# ================================
RESUME_FROM_CHECKPOINT = "./outputs/checkpoint-125" 
USE_CHECKPOINT = True  # Set to False to start fresh training

# ================================
# FORMAT TAGS
# ================================
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# ================================
# UTILITY FUNCTIONS
# ================================

def extract_solution(response):
    """Extracts the final solution from the text"""
    match = re.search(
        solution_start + r"(.*?)" + solution_end,
        response,
        flags=re.DOTALL
    )
    return match.group(1).strip() if match else None


def extract_reasoning(response):
    """Extracts the reasoning process from the text"""
    match = re.search(
        reasoning_start + r"(.*?)" + reasoning_end,
        response,
        flags=re.DOTALL
    )
    return match.group(1).strip() if match else ""


def normalize_answer(answer):
    """Normalizes an answer for comparison"""
    if answer is None:
        return ""
    
    answer = answer.strip().lower()
    answer = re.sub(r'[.,!?;:]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if numbers:
        return numbers[-1]
    return answer


def check_answer_correctness(solution, ground_truth):
    """Verifies if the solution is correct"""
    if solution is None or ground_truth is None:
        return False
    
    norm_sol = normalize_answer(solution)
    norm_gt = normalize_answer(ground_truth)
    
    if not norm_sol or not norm_gt:
        return False
    
    if norm_sol == norm_gt:
        return True
    
    try:
        sol_num = float(norm_sol)
        gt_num = float(norm_gt)
        return abs(sol_num - gt_num) < 1e-6
    except ValueError:
        pass
    
    return norm_sol in norm_gt or norm_gt in norm_sol


# ================================
# INTERNAL REWARD HEAD
# Self-Certainty Based
# ================================

def compute_self_certainty(model, tokenizer, prompt_msgs, response):
    """
    Internal Reward: Self-certainty of the model.
    Measures the model's confidence in its own predictions through token probabilities.
    """
    solution = extract_solution(response)
    
    if solution is None or len(solution.strip()) == 0:
        return 0.0
    
    try:
        full_conversation = prompt_msgs + [{"role": "assistant", "content": response}]
        input_ids = tokenizer.apply_chat_template(
            full_conversation,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Get solution tokens
            solution_start_idx = response.find(solution_start) + len(solution_start)
            solution_end_idx = response.find(solution_end)
            
            if solution_end_idx == -1:
                return 0.0
            
            solution_text = response[solution_start_idx:solution_end_idx].strip()
            solution_tokens = tokenizer.encode(solution_text, add_special_tokens=False)
            
            if len(solution_tokens) == 0:
                return 0.0
            
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            solution_token_start = None
            
            # Find solution position in sequence
            for i in range(len(response_tokens) - len(solution_tokens) + 1):
                if response_tokens[i:i+len(solution_tokens)] == solution_tokens:
                    for j in range(input_ids.shape[1] - len(solution_tokens) + 1):
                        if torch.all(input_ids[0, j:j+len(solution_tokens)] == torch.tensor(solution_tokens).to(input_ids.device)):
                            solution_token_start = j
                            break
                    if solution_token_start is not None:
                        break
            
            if solution_token_start is not None and solution_token_start > 0:
                solution_token_probs = []
                for i, token_id in enumerate(solution_tokens):
                    pos = solution_token_start + i
                    if pos < logits.shape[1] and pos > 0:
                        token_prob = probs[0, pos-1, token_id].item()
                        solution_token_probs.append(token_prob)
                
                if len(solution_token_probs) > 0:
                    # Calculate certainty as exponential of average log probability
                    log_probs = np.log(np.array(solution_token_probs) + 1e-10)
                    avg_log_prob = np.mean(log_probs)
                    certainty = np.exp(avg_log_prob)
                    return float(certainty)
        
        return 0.1
        
    except Exception as e:
        print(f"⚠️  Self-certainty calculation error: {e}")
        return 0.0


def compute_internal_reward(model, tokenizer, prompt_msgs, response):
    """
    INTERNAL REWARD HEAD
    Pure self-certainty based reward
    """
    certainty = compute_self_certainty(model, tokenizer, prompt_msgs, response)
    
    # Scale certainty to [0, 1] range
    internal_reward = certainty
    
    return internal_reward


# ================================
# EXTERNAL REWARD HEAD
# Format-Based + Answer-Based
# ================================

def compute_format_reward(response):
    """
    External Reward Component 1: Format-Based Evaluation
    Checks if the response follows the correct structure
    """
    score = 0.0
    
    # Check for presence of required tags
    has_reasoning_start = reasoning_start in response
    has_reasoning_end = reasoning_end in response
    has_solution_start = solution_start in response
    has_solution_end = solution_end in response
    
    # Full format compliance
    if has_reasoning_start and has_reasoning_end and has_solution_start and has_solution_end:
        score += 0.5
    else:
        return 0.0  # Invalid format gets zero
    
    # Extract sections
    reasoning = extract_reasoning(response)
    solution = extract_solution(response)
    
    # Check reasoning is not empty
    if reasoning and len(reasoning) > 10:
        score += 0.25
    
    # Check solution is not empty
    if solution and len(solution) > 0:
        score += 0.25
    
    return score


def compute_answer_reward(solution, ground_truth):
    """
    External Reward Component 2: Answer-Based Evaluation
    Checks if the answer is correct
    """
    is_correct = check_answer_correctness(solution, ground_truth)
    
    # Binary reward: 1.0 for correct, 0.0 for incorrect
    return 1.0 if is_correct else 0.0


def compute_external_reward(response, ground_truth):
    """
    EXTERNAL REWARD HEAD
    Combines format-based and answer-based evaluations
    
    Components:
    - Format Reward (40%): Structural correctness
    - Answer Reward (60%): Solution correctness
    """
    # Format evaluation
    format_score = compute_format_reward(response)
    
    # Answer evaluation
    solution = extract_solution(response)
    answer_score = compute_answer_reward(solution, ground_truth)
    
    # Weighted combination: 40% format + 60% answer
    external_reward = 0.4 * format_score + 0.6 * answer_score
    
    return external_reward


# ================================
# GLOBAL STATE
# ================================
class GlobalTrainingState:
    """Stores global state accessible to reward functions"""
    model = None
    tokenizer = None

global_state = GlobalTrainingState()


# ================================
# DUALHEAD REWARD FUNCTION
# 70% Internal + 30% External
# ================================

def compute_dualhead_reward(prompts, completions, answer, **kwargs):
    """
    DUALHEAD REWARD FUNCTION
    
    Combines two separate reward schemes:
    1. INTERNAL REWARD (70%): Self-certainty based
    2. EXTERNAL REWARD (30%): Format-based + Answer-based
    
    Final reward = 0.70 * Internal + 0.30 * External
    """
    scores = []
    
    model = global_state.model
    tokenizer = global_state.tokenizer
    
    # Weighting scheme
    INTERNAL_WEIGHT = 0.70
    EXTERNAL_WEIGHT = 0.30
    
    for prompt_msgs, completion, gt_answer in zip(prompts, completions, answer):
        # Extract response content
        if isinstance(completion, list) and len(completion) > 0:
            response = completion[0]["content"]
        elif isinstance(completion, dict):
            response = completion["content"]
        else:
            response = str(completion)
        
        # Basic validation
        solution = extract_solution(response)
        if solution is None or len(solution.strip()) == 0:
            scores.append(-10.0)  # Severe penalty for invalid format
            continue
        
        try:
            # ===========================
            # INTERNAL REWARD HEAD (70%)
            # ===========================
            internal_reward = compute_internal_reward(model, tokenizer, prompt_msgs, response)
            
            # ===========================
            # EXTERNAL REWARD HEAD (30%)
            # ===========================
            external_reward = compute_external_reward(response, gt_answer)
            
            # ===========================
            # DUALHEAD COMBINATION
            # ===========================
            final_reward = (
                INTERNAL_WEIGHT * internal_reward +
                EXTERNAL_WEIGHT * external_reward
            )
            
            # Optional: Bonus for high confidence + correct answer
            # (synergy between internal and external signals)
            if internal_reward > 0.8 and external_reward > 0.8:
                final_reward *= 1.2
            
        except Exception as e:
            print(f"⚠️  Reward calculation error: {e}")
            final_reward = -5.0
        
        scores.append(final_reward)
    
    # Batch-wise normalization
    scores = np.array(scores, dtype=np.float32)
    if len(scores) > 1:
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-9
        scores = (scores - mean_score) / std_score
        scores = np.clip(scores, -10.0, 10.0)
    
    return scores.tolist()


# ================================
# MODEL AND TOKENIZER
# ================================

if USE_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    checkpoint_path = RESUME_FROM_CHECKPOINT
    print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
else:
    checkpoint_path = "./models/SFT_Base_Model"
    print(f"🆕 Starting new training from: {checkpoint_path}")

max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.75,
)

tokenizer.padding_side = 'left'

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

# Store in global state
global_state.model = model
global_state.tokenizer = tokenizer

max_prompt_length = 512
max_completion_length = 1536

model.generation_config.max_new_tokens = max_completion_length
model.generation_config.max_length = max_seq_length
model.generation_config.temperature = 0.85
model.generation_config.top_p = 1.0
model.generation_config.top_k = -1
model.generation_config.do_sample = True
model.generation_config.bos_token_id = tokenizer.bos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

print("✅ Model loaded successfully with Unsloth!")

# ================================
# DATASET
# ================================
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
dataset = dataset.shuffle(seed=3407).select(range(1000))

dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["prompt"]},
    ],
    "answer": x["solution"],
})

tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)

# ================================
# vLLM Sampling
# ================================
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[
        reasoning_end,
        solution_end,
        "</SOLUTION>",
        "</SOLUTION>\n",
        "</SOLUTION> ",
        tokenizer.eos_token
    ],
    include_stop_str_in_output=True,
)

# ================================
# GRPO TRAINING
# ================================
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-7,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,
    save_steps=25,
    output_dir="./outputs_GRPO_dualhead",
    report_to="none",
    save_total_limit=3,
    remove_unused_columns=False,
)

print("🎭 DUALHEAD MODE: Two Separate Reward Schemes")
print("=" * 60)
print("📊 REWARD ARCHITECTURE:")
print()
print("   🧠 INTERNAL HEAD (70% weight):")
print("      └─ Self-Certainty: Model's confidence in predictions")
print()
print("   🎯 EXTERNAL HEAD (30% weight):")
print("      ├─ Format-Based (40%): Structural correctness")
print("      └─ Answer-Based (60%): Solution correctness")
print()
print("   ⚖️  Final Reward = 0.70 × Internal + 0.30 × External")
print()
print("   🎁 Synergy Bonus: 1.2× when both heads > 0.8")
print("=" * 60)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[compute_dualhead_reward],
    args=training_args,
    train_dataset=dataset,
)

model.gradient_checkpointing_enable()
gc.collect()
torch.cuda.empty_cache()

print("🚀 Starting GRPO DualHead training...")

# Training
if USE_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    print(f"📂 Resuming from: {RESUME_FROM_CHECKPOINT}")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
else:
    trainer.train()

# Save final model
output_path = "./models/Qwen_GRPO_DualHead"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n🏁 DualHead Training complete!")
print(f"💾 Model saved to {output_path}")
print(f"🎭 Model trained with balanced internal + external rewards!")
print(f"📈 70% self-certainty + 30% objective evaluation")