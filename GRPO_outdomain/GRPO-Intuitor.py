# ==============================================
# GRPO training script - PURE INTUITOR STYLE
# Internal Confidence Only (No Structure Rewards)
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


# ================================
# PURE CONFIDENCE COMPUTATION
# ================================

def compute_sequence_confidence(model, tokenizer, prompt_msgs, response):
    """
    Computes the model's internal confidence for the ENTIRE response.
    Pure self-confidence: average log probability of all generated tokens.
    """
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
            
            # Encode the full response
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            if len(response_tokens) == 0:
                return 0.0
            
            # Find where the response starts in the full sequence
            full_tokens = input_ids[0].cpu().tolist()
            response_start = None
            
            for i in range(len(full_tokens) - len(response_tokens) + 1):
                if full_tokens[i:i+len(response_tokens)] == response_tokens:
                    response_start = i
                    break
            
            if response_start is None or response_start == 0:
                return 0.0
            
            # Collect probabilities for each token in the response
            token_probs = []
            for i, token_id in enumerate(response_tokens):
                pos = response_start + i
                if pos < logits.shape[1] and pos > 0:
                    token_prob = probs[0, pos-1, token_id].item()
                    token_probs.append(token_prob)
            
            if len(token_probs) == 0:
                return 0.0
            
            # Calculate average log probability
            log_probs = np.log(np.array(token_probs) + 1e-10)
            avg_log_prob = np.mean(log_probs)
            
            # Convert to confidence score (exponential of avg log prob)
            confidence = np.exp(avg_log_prob)
            
            return float(confidence)
        
    except Exception as e:
        print(f"⚠️  Confidence calculation error: {e}")
        return 0.0


def compute_solution_confidence(model, tokenizer, prompt_msgs, response):
    """
    Computes confidence specifically for the solution section.
    This gives higher weight to the final answer's confidence.
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
            
            solution_tokens = tokenizer.encode(solution, add_special_tokens=False)
            
            if len(solution_tokens) == 0:
                return 0.0
            
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            solution_token_start = None
            
            # Find solution tokens in response
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
                    log_probs = np.log(np.array(solution_token_probs) + 1e-10)
                    avg_log_prob = np.mean(log_probs)
                    confidence = np.exp(avg_log_prob)
                    return float(confidence)
        
        return 0.0
        
    except Exception as e:
        print(f"⚠️  Solution confidence error: {e}")
        return 0.0


def compute_entropy_based_confidence(model, tokenizer, prompt_msgs, response):
    """
    Measures confidence through entropy of the probability distribution.
    Lower entropy = model is more certain about its predictions.
    """
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
            
            # Calculate entropy for each position
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            # Average entropy across the sequence
            avg_entropy = entropy.mean().item()
            
            # Convert entropy to confidence score (inverse relationship)
            # Normalize by max possible entropy (log of vocab size)
            max_entropy = np.log(probs.shape[-1])
            normalized_entropy = avg_entropy / max_entropy
            
            # Confidence is inverse of normalized entropy
            confidence = 1.0 - normalized_entropy
            
            return float(max(0.0, confidence))
        
    except Exception as e:
        print(f"⚠️  Entropy confidence error: {e}")
        return 0.0


# ================================
# GLOBAL STATE
# ================================
class GlobalTrainingState:
    """Stores global state accessible to reward functions"""
    model = None
    tokenizer = None

global_state = GlobalTrainingState()


# ================================
# PURE CONFIDENCE REWARD FUNCTION
# ================================

def compute_pure_confidence_reward(prompts, completions, answer, **kwargs):
    """
    PURE CONFIDENCE REWARD: Only internal model confidence.
    
    Components:
    1. Solution Confidence (60%) - Confidence in the final answer
    2. Sequence Confidence (30%) - Confidence in the entire response
    3. Entropy-based Confidence (10%) - Low entropy = high confidence
    
    NO external validation, NO structure checking.
    Pure self-belief signals only.
    """
    scores = []
    
    model = global_state.model
    tokenizer = global_state.tokenizer
    
    for prompt_msgs, completion in zip(prompts, completions):
        # Extract response content
        if isinstance(completion, list) and len(completion) > 0:
            response = completion[0]["content"]
        elif isinstance(completion, dict):
            response = completion["content"]
        else:
            response = str(completion)
        
        # Minimal format check (just to ensure parseable response)
        solution = extract_solution(response)
        if solution is None or len(solution.strip()) == 0:
            scores.append(-10.0)  # Severe penalty for unparseable format
            continue
        
        try:
            # === COMPONENT 1: Solution Confidence (60%) ===
            # The model's belief in its final answer
            solution_conf = compute_solution_confidence(model, tokenizer, prompt_msgs, response)
            
            # === COMPONENT 2: Sequence Confidence (30%) ===
            # Overall confidence across the entire response
            sequence_conf = compute_sequence_confidence(model, tokenizer, prompt_msgs, response)
            
            # === COMPONENT 3: Entropy Confidence (10%) ===
            # Low entropy = high decisiveness/confidence
            entropy_conf = compute_entropy_based_confidence(model, tokenizer, prompt_msgs, response)
            
            # === FINAL REWARD (Pure Weighted Confidence) ===
            final_reward = (
                0.60 * solution_conf +
                0.30 * sequence_conf +
                0.10 * entropy_conf
            )
            
            # Amplify very high confidence (model is very sure)
            if solution_conf > 0.8 and sequence_conf > 0.7:
                final_reward *= 1.5
            
            # No penalty for low confidence - let the model explore
            # Natural selection will favor confident responses
            
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
    "answer": x["solution"],  # Not used in reward, kept for compatibility
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
    output_dir="./outputs_GRPO_pure_confidence",
    report_to="none",
    save_total_limit=3,
    remove_unused_columns=False,
)

print("🧠 PURE CONFIDENCE MODE: Internal Self-Belief Only")
print("📊 Reward Components:")
print("   - 60% Solution Confidence (belief in final answer)")
print("   - 30% Sequence Confidence (belief in full response)")
print("   - 10% Entropy-based Confidence (decisiveness)")
print("⚡ ZERO external signals - pure introspective learning!")
print("🎯 Bonus: 1.5x multiplier when solution_conf > 0.8 AND sequence_conf > 0.7")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[compute_pure_confidence_reward],
    args=training_args,
    train_dataset=dataset,
)

model.gradient_checkpointing_enable()
gc.collect()
torch.cuda.empty_cache()

print("🚀 Starting GRPO Pure Confidence training...")

# Training
if USE_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    print(f"📂 Resuming from: {RESUME_FROM_CHECKPOINT}")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
else:
    trainer.train()

# Save final model
output_path = "./models/Qwen_GRPO_Intuitor"  # Insert desired output path here
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"\n🏁 Pure Confidence Training complete!")
print(f"💾 Model saved to {output_path}")
print(f"🔬 Model trained purely on internal confidence signals!")
print(f"💭 The model learned to trust its own intuition!")