import torch
import json
import csv
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Import classes from your original file GRPO_BFCL_train.py
# (Ensure the filename matches your actual module name)
from GRPO_BFCL_train import BFCLDataset, BFCLEvaluator

def format_prompt_eval(example):
    """Maintains the exact same format used during training"""
    prompt = "You are a function-calling assistant. Given a user query and system state, you must identify and call ALL necessary functions to complete the request.\n\n"
    prompt += "="*60 + "\nSYSTEM STATE:\n" + "="*60 + "\n"
    try:
        config = json.loads(example['initial_config'])
        for key, value in config.items():
            if isinstance(value, dict):
                prompt += f"\n{key}:\n"
                for k, v in value.items(): prompt += f"  {k}: {v}\n"
            else: prompt += f"{key}: {value}\n"
    except:
        prompt += example['initial_config'][:500] + "...\n"
    
    prompt += "\n" + "="*60 + "\nUSER QUERY:\n" + "="*60 + "\n"
    prompt += example['query'] + "\n\n"
    prompt += "="*60 + "\nAVAILABLE FUNCTIONS:\n" + "="*60 + "\n"
    
    for func in example['available_functions']:
        desc = func.get('description', 'No description')
        prompt += f"  • {func['name']}: {desc}\n"
    
    prompt += "\n" + "─"*60 + "\nINSTRUCTIONS:\n" + "─"*60 + "\n"
    prompt += "1. Analyze the query and system state\n2. Identify ALL functions needed\n3. Provide output as JSON array inside <SOLUTION> tag.\n\n"
    prompt += "YOUR RESPONSE:\n" + "="*60 + "\n"
    return prompt

def run_evaluation():
    # --- PATH CONFIGURATION ---
    BASE_MODEL_NAME = "./models/GRPO_Hybrid_Base"
    LORA_ADAPTER_PATH = "./models/BFCL_Qwen_Final" 
    TEST_DATA_PATH = "./dataset/test.json"
    OUTPUT_CSV = "evaluation_results.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- Starting Model Evaluation ---")
    
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
    tokenizer.padding_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()

    dataset = BFCLDataset(TEST_DATA_PATH, tokenizer)
    evaluator = BFCLEvaluator()

    results = []
    perfect_matches = 0 # Counter for Reward >= 0.95

    print(f"Running on {len(dataset)} examples from the Test Set...\n")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            example = dataset[i]
            prompt = format_prompt_eval(example)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, 
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            reward = evaluator.compute_reward(
                generated_text, 
                example['ground_truth'], 
                example['available_functions']
            )
            
            # Calculate Perfect Match
            if reward >= 0.95:
                perfect_matches += 1
            
            extracted_calls = evaluator.extract_function_calls(generated_text)
            gt_calls = [f"{gt['name']}" for gt in example['ground_truth']]
            pred_calls = [f"{pred['name']}" for pred in extracted_calls]

            results.append({
                "id": i,
                "query": example['query'][:100].replace("\n", " "),
                "ground_truth": "|".join(gt_calls),
                "prediction": "|".join(pred_calls),
                "reward": round(reward, 4),
                "full_response": generated_text.replace("\n", " ")
            })

    # --- CSV WRITING ---
    if results:
        keys = results[0].keys()
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    # --- FINAL STATISTICS CALCULATION ---
    total_samples = len(results)
    if total_samples > 0:
        avg_reward = sum(r['reward'] for r in results) / total_samples
        perfect_match_rate = (perfect_matches / total_samples) * 100

        print(f"\n" + "="*40)
        print(f"FINAL EVALUATION REPORT")
        print(f"="*40)
        print(f"Total samples:       {total_samples}")
        print(f"Average Reward:      {avg_reward:.4f}")
        print(f"Perfect Matches:     {perfect_matches}")
        print(f"Perfect Match Rate:  {perfect_match_rate:.2f}%")
        print(f"="*40)
        print(f"Results saved to:    {OUTPUT_CSV}\n")
    else:
        print("No results to display.")

if __name__ == "__main__":
    run_evaluation()