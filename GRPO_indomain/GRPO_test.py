"""
Test script for the fine-tuned GRPO model
Loads the model and generates responses for sample mathematical problems
"""

from unsloth import FastLanguageModel
import torch

# ========== Configuration ==========
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# ========== Model Loading ==========
# Use the latest checkpoint or the saved model
checkpoint_path = "/home/jovyan/Toy_Problem_v2/models/Qwen_GRPO_1000"  # Update with your checkpoint
# or
# checkpoint_path = "/home/jovyan/Toy_Problem_v2/models/GRPO_Qwen_SFT_1000"

print(f"Loading model from: {checkpoint_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# Enable fast inference mode
FastLanguageModel.for_inference(model)

print("Model loaded successfully!\n")

# ========== Test Problems ==========
test_problems = [
    "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    "What is 15% of 240?",
    "A rectangle has a length of 12 cm and width of 8 cm. What is its area?",
    "If x + 5 = 12, what is the value of x?",
    "A store sells apples for $0.50 each. If you buy 24 apples, how much do you pay?",
]

# ========== Inference Function ==========
def generate_solution(problem):
    """Generates a solution for a given problem"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1536,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return response

# ========== Problem Testing ==========
print("=" * 80)
print("GRPO MODEL TESTING")
print("=" * 80)

for i, problem in enumerate(test_problems, 1):
    print(f"\n{'=' * 80}")
    print(f"PROBLEM {i}")
    print(f"{'=' * 80}")
    print(f"Question: {problem}\n")
    
    try:
        solution = generate_solution(problem)
        print(f"Model response:\n{solution}")
        
        # Format check
        has_reasoning = reasoning_start in solution and reasoning_end in solution
        has_solution = solution_start in solution and solution_end in solution
        
        print(f"\n✓ Format check:")
        print(f"  - Reasoning tags: {'✓' if has_reasoning else '✗'}")
        print(f"  - Solution tags: {'✓' if has_solution else '✗'}")
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
    
    print(f"{'=' * 80}\n")

# ========== Interactive Mode (optional) ==========
print("\n" + "=" * 80)
print("INTERACTIVE MODE")
print("=" * 80)
print("Type 'quit' to exit\n")

while True:
    user_problem = input("Enter a math problem: ").strip()
    
    if user_problem.lower() in ['quit', 'exit', 'q']:
        print("Exiting...")
        break
    
    if not user_problem:
        continue
    
    print("\nGenerating response...\n")
    try:
        solution = generate_solution(user_problem)
        print(f"Response:\n{solution}\n")
    except Exception as e:
        print(f"Error: {e}\n")

print("\nTesting complete!")