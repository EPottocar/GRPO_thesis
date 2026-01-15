import torch
import gc
import shutil
from torch.utils.data import Dataset, DataLoader
#from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import json
import re
import ast
from typing import Dict, List, Tuple, Any
import copy
import time
import random

def print_mem(tag):
    if torch.cuda.is_available():
        print(f"[MEM] {tag} | allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB")


class BFCLDataset(Dataset):
    """Dataset for BFCL benchmark - Modified for your format"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.examples = []
        for item in data:
            # Extract function calls from ground_truth
            ground_truth_calls = []
            for gt_call in item.get('ground_truth', []):
                # Parse function call: "function_name(param1=val1, param2=val2)"
                func_info = self._parse_function_call(gt_call)
                if func_info:
                    ground_truth_calls.append(func_info)
            
            # Extract available functions from initial_config and ground_truth
            available_functions = self._extract_available_functions(item)
            
            self.examples.append({
                'query': item['query'],
                'initial_config': item.get('initial_config', '{}'),
                'involved_classes': item.get('involved_classes', []),
                'ground_truth': ground_truth_calls,
                'available_functions': available_functions
            })
    
    def _parse_function_call(self, call_str: str) -> Dict[str, Any]:
        """Parse a function call string like 'func_name(param1=val1, param2=val2)'"""
        try:
            # Extract function name
            match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', call_str)
            if not match:
                return {'name': call_str.strip(), 'parameters': {}}
            
            func_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            parameters = {}
            if params_str.strip():
                # Handle nested structures by using ast.literal_eval
                try:
                    # Create a temporary dict to evaluate
                    temp_dict_str = f"dict({params_str})"
                    parameters = ast.literal_eval(temp_dict_str)
                except:
                    # Fallback: simple regex parsing
                    param_pattern = r'(\w+)\s*=\s*([^,]+)'
                    for match in re.finditer(param_pattern, params_str):
                        key = match.group(1)
                        value = match.group(2).strip()
                        # Try to evaluate the value
                        try:
                            parameters[key] = ast.literal_eval(value)
                        except:
                            parameters[key] = value
            
            return {
                'name': func_name,
                'parameters': parameters
            }
        except Exception as e:
            print(f"Error parsing function call '{call_str}': {e}")
            return {'name': call_str.strip(), 'parameters': {}}
    
    def _extract_available_functions(self, item: Dict) -> List[Dict]:
        """Extract all unique functions from ground_truth and involved_classes"""
        functions = {}
        
        # From ground_truth
        for gt_call in item.get('ground_truth', []):
            func_info = self._parse_function_call(gt_call)
            func_name = func_info['name']
            if func_name not in functions:
                functions[func_name] = {
                    'name': func_name,
                    'description': f"Function from {item.get('involved_classes', ['API'])[0] if item.get('involved_classes') else 'API'}",
                    'parameters': func_info.get('parameters', {})
                }
        
        return list(functions.values())
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class BFCLEvaluator:
    """BFCL Evaluation Logic with <reasoning> and <tool> tag support"""
    
    @staticmethod
    def extract_function_calls(completion: Any) -> List[Dict[str, Any]]:
        """Extract function calls from JSON array/object inside <SOLUTION> tags"""
        function_calls = []
        seen_functions = set()
        
        # 1. ROBUSTNESS CHECK: Ensure completion is a string
        if not isinstance(completion, str):
            # Se è una lista (output comune dei modelli), prova a prendere il primo elemento
            if isinstance(completion, list) and completion:
                # Tentativo di prendere il primo elemento come stringa
                if isinstance(completion[0], str):
                    completion = completion[0]
                else:
                    # Fallback a stringa per l'intera lista
                    completion = str(completion) 
            else:
                # Fallback per oggetti sconosciuti
                completion = str(completion)
        
        # 2. Extract the content within <SOLUTION>...</SOLUTION>
        solution_pattern = r'<SOLUTION>\s*(.*?)\s*</SOLUTION>'
        solution_match = re.search(solution_pattern, completion, re.DOTALL | re.IGNORECASE)
        
        content_to_parse = None
        if solution_match:
            content_to_parse = solution_match.group(1).strip()
        else:
            # Fallback: if no solution tag found, look for JSON array/object anywhere
            content_to_parse = completion 

        # 3. Try to parse the content as JSON (array of calls or single object)
        if content_to_parse:
            try:
                parsed = json.loads(content_to_parse)
                
                # If single JSON object is returned, treat it as a list with one element
                if isinstance(parsed, dict) and 'name' in parsed:
                    parsed = [parsed]
                
                if isinstance(parsed, list):
                    # Array of JSON objects
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            func_name = item['name']
                            if func_name not in seen_functions:
                                function_calls.append({
                                    'name': func_name,
                                    'parameters': item.get('parameters', {})
                                })
                                seen_functions.add(func_name)
            except json.JSONDecodeError:
                # If JSON parsing fails (e.g., incomplete output), try to find "name" fields via regex
                name_pattern = r'"name"\s*:\s*"([^"]+)"'
                names = re.findall(name_pattern, content_to_parse)
                for name in names:
                    if name not in seen_functions:
                        function_calls.append({'name': name, 'parameters': {}})
                        seen_functions.add(name)
        
        return function_calls
    
    @staticmethod
    def compute_reward(completion: str, ground_truth: List[Dict], available_functions: List[Dict]) -> float:
        """
        Compute reward based on:
        1. Coverage of required functions (primary)
        2. Format correctness (bonus) - checking <start_working_out> and <SOLUTION>
        3. Avoiding duplicates and invalid functions (penalties)
        """
        try:
            # Extract function calls from completion
            extracted = BFCLEvaluator.extract_function_calls(completion)
            
            if not extracted:
                return 0.0
            
            # Get ground truth function names
            gt_names = {gt['name'] for gt in ground_truth}
            completion_names = {call['name'] for call in extracted}
            available_names = {f['name'] for f in available_functions}
            
            # 1. MAIN REWARD: Coverage of required functions
            if not gt_names:
                base_reward = 0.0
            else:
                correct = gt_names.intersection(completion_names)
                base_reward = len(correct) / len(gt_names)
            
            # 2. FORMAT BONUS
            # Check for the required tags
            has_reasoning = '<start_working_out>' in completion and '</end_working_out>' in completion
            has_solution = '<SOLUTION>' in completion and '</SOLUTION>' in completion
            
            # The tool tag check is removed, only reasoning and solution tags are required
            properly_formatted = has_reasoning and has_solution
            format_bonus = 0.05 if properly_formatted else 0.0
            
            # 3. COMPLETENESS BONUS: All required functions called
            completeness_bonus = 0.1 if gt_names.issubset(completion_names) else 0.0
            
            # 4. PENALTY: Invalid functions (not in available functions)
            invalid_functions = completion_names - available_names
            invalid_penalty = len(invalid_functions) * 0.1
            
            # 5. PENALTY: Excessive duplicates
            func_counts = {}
            for f in extracted:
                func_counts[f['name']] = func_counts.get(f['name'], 0) + 1
            
            duplicate_penalty = 0.0
            for count in func_counts.values():
                if count >= 3:
                    duplicate_penalty += 0.05 * (count - 2)
            
            # 6. PENALTY: Calling too many extra functions
            extra_functions = completion_names - gt_names
            extra_penalty = len(extra_functions) * 0.03
            
            # Total reward
            reward = base_reward + format_bonus + completeness_bonus
            reward -= (invalid_penalty + duplicate_penalty + extra_penalty)
            reward = max(0.0, min(1.0, reward))
            
            return reward
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0


class GRPOTrainer:
    """Group Relative Policy Optimization Trainer with QLoRA"""
    def __init__(
        self,
        model_name: str,                 
        train_data_path: str,
        test_data_path: str,
        output_dir: str,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 1e-6,
        max_grad_norm: float = 0.2,
        kl_beta: float = 0.001,
        ref_update_steps: int = 100,
        max_completion_length: int = 512,
        num_epochs: int = 100,
        warmup_steps: int = 20,
        mu: int = 2,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        device: str = "cuda",
        checkpoint_path: str = None,
        num_completions_per_prompt: int = 4
    ):
        self.device = device
        self.output_dir = output_dir
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        
        # --- Batch & optimizer params ---
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = per_device_batch_size * gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.kl_beta = kl_beta
        self.ref_update_steps = ref_update_steps
        self.max_completion_length = max_completion_length
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.num_completions_per_prompt = num_completions_per_prompt

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # --- Tokenizer ---
        print(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # --- Base model ---
        print(f"Loading base model from {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # --- Load adapter (LoRA) if presente ---
        adapter_path = os.path.join(model_name)
        if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
            print("Loading adapter weights...")
            self.policy_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=True
            )
        else:
            print("No adapter found, using base model as policy_model")
            self.policy_model = base_model

        self.policy_model.print_trainable_parameters()
        
        # --- Free base_model memory if needed ---
        try:
            del base_model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # --- Reference model inizializzato ---
        self.ref_model = None
        
        # --- Datasets ---
        self.train_dataset = BFCLDataset(train_data_path, self.tokenizer)
        self.test_dataset = BFCLDataset(test_data_path, self.tokenizer)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=per_device_batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )
        
        # --- Optimizer ---
        trainable_params = filter(lambda p: p.requires_grad, self.policy_model.parameters())
        self.optimizer = AdamW(trainable_params, lr=learning_rate)
        
        # --- Scheduler ---
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        self.global_step = 0
        self.replay_buffer = []
        self.evaluator = BFCLEvaluator()

        # NUOVO: Early stopping
        self.best_test_reward = 0.0
        self.patience = 5
        self.patience_counter = 0
        self.best_model_path = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Carica checkpoint se presente ---
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.global_step = state.get('global_step', 0)
                    print(f"✓ Resuming from global step: {self.global_step}")
            
            if os.path.exists(optimizer_path):
                try:
                    self.optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
                    print("✓ Loaded optimizer state.")
                except Exception as e:
                    print(f"⚠ Could not load optimizer state: {e}")
            
            if os.path.exists(scheduler_path):
                try:
                    self.scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
                    print("✓ Loaded scheduler state.")
                except Exception as e:
                    print(f"⚠ Could not load scheduler state: {e}")
        else:
            print("Starting fresh training (no checkpoint found)")
        
        # --- Initialize reference model ---
        self.update_reference_model(initial_load=True)

    
    def update_reference_model(self, initial_load=False):
        print(f"Updating reference model at step {self.global_step}")

        if self.ref_model is None:
            print("Loading reference base model once...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                device_map="auto"
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        # salva solo adapter LoRA
        tmp_path = os.path.join(self.output_dir, "tmp_ref_update")
        self.policy_model.save_pretrained(tmp_path)

        if not isinstance(self.ref_model, PeftModel):
            self.ref_model = PeftModel.from_pretrained(
                self.ref_model,
                tmp_path,
                is_trainable=False
            )
        else:
            self.ref_model.load_adapter(tmp_path, adapter_name="ref")
            self.ref_model.set_adapter("ref")

        self.ref_model.eval()
        shutil.rmtree(tmp_path)

    
    def train(self):
        """Main training loop with early stopping"""
        print("Starting GRPO training with QLoRA...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Output directory: {self.output_dir}")
        
        self.policy_model.train()
        start_training_time = time.time()
        
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        start_epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        if start_epoch > 0:
            print(f"✓ Resuming from epoch {start_epoch + 1}")
    
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_batches = 0
            accumulated_batch = []
    
            for batch_idx, example in enumerate(self.train_loader):
                accumulated_batch.extend(example)
                
                if len(accumulated_batch) >= self.effective_batch_size or \
                   batch_idx == len(self.train_loader) - 1:
    
                    if len(accumulated_batch) < self.per_device_batch_size and len(self.train_loader) > 1:
                        accumulated_batch = []
                        continue
    
                    step_start_time = time.time()
                    loss, rewards = self.train_step(accumulated_batch, self.global_step)
                    step_end_time = time.time()
    
                    step_time = step_end_time - step_start_time
                    total_elapsed = step_end_time - start_training_time
                    steps_done = self.global_step + 1
                    total_steps = self.num_epochs * len(self.train_loader) // self.gradient_accumulation_steps
                    eta_seconds = step_time * (total_steps - steps_done)
    
                    if steps_done % 10 == 0:
                        print(f"[Epoch {epoch+1}/{self.num_epochs} | Step {steps_done}/{total_steps}] "
                              f"Loss: {loss:.4f}, Avg Reward: {sum(rewards)/len(rewards):.3f}, "
                              f"Step Time: {step_time:.2f}s, Elapsed: {total_elapsed/60:.2f}min, "
                              f"ETA: {eta_seconds/60:.2f}min")
    
                    if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
    
                    epoch_loss += loss
                    epoch_reward += sum(rewards) / len(rewards)
                    num_batches += 1
    
                    if self.global_step > 0 and (self.global_step + 1) % self.ref_update_steps == 0:
                        self.update_reference_model()
    
                    self.global_step += 1
                    accumulated_batch = []
    
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            avg_epoch_reward = epoch_reward / max(num_batches, 1)
            total_elapsed_epoch = time.time() - start_training_time
    
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"{'='*80}")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Average Reward: {avg_epoch_reward:.3f}")
            print(f"  Total Elapsed Time: {total_elapsed_epoch/60:.2f}min")
            print(f"{'='*80}\n")
    
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")
    
            # NUOVO: Valutazione e early stopping
            test_metrics = self.evaluate_test_set()
            test_reward = test_metrics['avg_reward']
            
            print(f"  *** Test Metrics ***")
            print(f"  Avg Reward: {test_reward:.4f}")
            print(f"  Perfect Match Rate: {test_metrics['perfect_match_rate']:.4f}")
            print(f"  Best Test Reward So Far: {self.best_test_reward:.4f}\n")
            
            # NUOVO: Track best model
            if test_reward > self.best_test_reward:
                self.best_test_reward = test_reward
                self.patience_counter = 0
                
                # Salva il best model
                best_path = os.path.join(self.output_dir, "best_model")
                if os.path.exists(best_path):
                    shutil.rmtree(best_path)
                shutil.copytree(
                    os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}"),
                    best_path
                )
                self.best_model_path = best_path
                print(f"  ✓ New best model saved! (reward: {test_reward:.4f})")
            else:
                self.patience_counter += 1
                print(f"  ⚠ No improvement for {self.patience_counter} epoch(s)")
                
                # NUOVO: Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING triggered after {self.patience} epochs without improvement")
                    print(f"Best test reward: {self.best_test_reward:.4f}")
                    print(f"Best model saved in: {self.best_model_path}")
                    print(f"{'='*80}\n")
                    break
    
        total_training_time = time.time() - start_training_time
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total training time: {total_training_time/60:.2f} minutes")
        print(f"Best test reward: {self.best_test_reward:.4f}")
        print(f"Best model location: {self.best_model_path}")
        
        # Copia il best model come final_model
        final_path = os.path.join(self.output_dir, "final_model")
        if self.best_model_path and os.path.exists(self.best_model_path):
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(self.best_model_path, final_path)
            print(f"✓ Best model copied to: {final_path}")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint"""
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps if self.gradient_accumulation_steps > 0 else 1
        current_epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        trainer_state = {
            "global_step": self.global_step,
            "epoch": current_epoch,
            "replay_buffer_size": len(self.replay_buffer),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate
        }
        with open(os.path.join(save_path, "trainer_state.json"), 'w') as f:
            json.dump(trainer_state, f, indent=2)
        
        print(f"✓ Checkpoint saved to {save_path}")

    def format_prompt(self, example: Dict) -> str:
        """Format prompt with query, initial config, and available functions"""
        
        prompt = "You are a function-calling assistant. Given a user query and system state, you must identify and call ALL necessary functions to complete the request.\n\n"
        
        # Add initial config (system state)
        prompt += "="*60 + "\n"
        prompt += "SYSTEM STATE:\n"
        prompt += "="*60 + "\n"
        try:
            config = json.loads(example['initial_config'])
            # Format config nicely
            for key, value in config.items():
                if isinstance(value, dict):
                    prompt += f"\n{key}:\n"
                    for k, v in value.items():
                        prompt += f"  {k}: {v}\n"
                else:
                    prompt += f"{key}: {value}\n"
        except:
            prompt += example['initial_config'][:500] + "...\n"
        
        prompt += "\n" + "="*60 + "\n"
        prompt += "USER QUERY:\n"
        prompt += "="*60 + "\n"
        prompt += example['query'] + "\n\n"
        
        prompt += "="*60 + "\n"
        prompt += "AVAILABLE FUNCTIONS:\n"
        prompt += "="*60 + "\n"
        
        for func in example['available_functions']:
            desc = func.get('description', 'No description')
            prompt += f"  • {func['name']}: {desc}\n"
        
        prompt += "\n" + "─"*60 + "\n"
        prompt += "INSTRUCTIONS:\n"
        prompt += "─"*60 + "\n"
        prompt += "1. Analyze the query and system state\n"
        prompt += "2. Identify ALL functions needed to fulfill the request\n"
        prompt += "3. Provide the output as a single JSON array of function calls inside the <SOLUTION> tag.\n"
        prompt += "4. The JSON format must include 'name' and 'parameters' fields for each call.\n\n"
        
        prompt += "OUTPUT FORMAT:\n"
        prompt += "─"*60 + "\n"
        prompt += "1. <start_working_out>Explain what functions are needed and why</end_working_out>\n"
        prompt += "2. <SOLUTION>\n"
        prompt += '3. [{"name": "function_name1", "parameters": {...}}, {"name": "function_name2", "parameters": {...}}]\n'
        prompt += "4. </SOLUTION>\n\n"
        
        prompt += "EXAMPLE:\n"
        prompt += "─"*60 + "\n"
        prompt += "Query: Start the car engine\n"
        prompt += "<start_working_out>To start the engine, I need to first press the brake pedal, then use the START ignition mode</end_working_out>\n"
        prompt += "<SOLUTION>\n"
        prompt += '[{"name": "pressBrakePedal", "parameters": {"pedalPosition": 1.0}}, {"name": "startEngine", "parameters": {"ignitionMode": "START"}}]\n'
        prompt += "</SOLUTION>\n\n"
        
        prompt += "="*60 + "\n"
        prompt += "YOUR RESPONSE:\n"
        prompt += "="*60 + "\n"
        
        return prompt
    
    def generate_completions(self, prompts: List[str], model, num_completions: int = 1) -> Tuple[List[List[str]],List[torch.Tensor]]:
        """
        Generate multiple completions per prompt
        
        Args:
            prompts: List of prompts
            model: Model to use for generation  
            num_completions: Number of completions to generate per prompt
            
        Returns:
            Tuple of (completions_per_prompt, sequences_per_prompt)
            completions_per_prompt: List[List[str]] - outer list = prompts, inner = completions
        """
        all_completions_per_prompt = []
        all_sequences_per_prompt = []
        
        for prompt in prompts:
            # Replicate prompt N times for parallel generation
            inputs = self.tokenizer(
                [prompt] * num_completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            model_device = next(model.parameters()).device
            for k, v in inputs.items():
                inputs[k] = v.to(model_device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    repetition_penalty=1.15,
                )
            
            completions = self.tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Process completions (limit tool calls)
            MAX_TOOLS = 6
            processed = []
            for comp in completions:
                comp_lower = comp.lower()
                tool_count = comp_lower.count('</tool>')
                
                if tool_count > MAX_TOOLS:
                    pos = 0
                    for i in range(MAX_TOOLS):
                        next_tool_end = comp_lower.find('</tool>', pos)
                        if next_tool_end == -1:
                            break
                        pos = next_tool_end + len('</tool>')
                    comp = comp[:pos]
                
                processed.append(comp)
            
            all_completions_per_prompt.append(processed)
            all_sequences_per_prompt.append(outputs.sequences)
        
        return all_completions_per_prompt, all_sequences_per_prompt
    
    def compute_rewards(self, examples: List[Dict], completions: List[str]) -> List[float]:
        """Compute rewards using the evaluator"""
        rewards = []
        for example, completion in zip(examples, completions):
            reward = self.evaluator.compute_reward(
                completion,
                example['ground_truth'],
                example['available_functions']
            )
            rewards.append(float(reward))
        return rewards
    
    def compute_log_probs(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for given sequences"""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=2,
                index=labels.unsqueeze(2)
            ).squeeze(2)
            
            mask = attention_mask[:, 1:].bool()
            token_log_probs = token_log_probs * mask
            
            return token_log_probs.sum(dim=1)

    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Evaluate on test set - use the BEST completion among the N generated
        """
        self.policy_model.eval()
        
        total_reward = 0
        perfect_matches = 0
        total_examples = 0
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_device_batch_size,
            collate_fn=lambda x: x 
        )
        
        with torch.no_grad():
            for batch_examples in test_loader:
                prompts = [self.format_prompt(ex) for ex in batch_examples]
                
                # Generate multiple completions
                completions_per_prompt, _ = self.generate_completions(
                    prompts, 
                    self.policy_model,
                    num_completions=self.num_completions_per_prompt
                )
                
                # Take BEST completion per prompt
                for i, example in enumerate(batch_examples):
                    completions_for_this = completions_per_prompt[i]
                    rewards_for_this = self.compute_rewards(
                        [example] * len(completions_for_this),
                        completions_for_this
                    )
                    
                    best_reward = max(rewards_for_this)
                    total_reward += best_reward
                    perfect_matches += 1 if best_reward >= 0.95 else 0
                    total_examples += 1
                    
        self.policy_model.train()
        
        return {
            'avg_reward': total_reward / max(total_examples, 1),
            'perfect_match_rate': perfect_matches / max(total_examples, 1)
        }
    
    def grpo_loss(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        is_on_policy: bool = True
    ) -> torch.Tensor:
        """
        Compute GRPO loss with GROUP-LEVEL normalization
        
        Tutti i prompts in input dovrebbero essere identici (stesso gruppo)
        per normalizzare correttamente gli advantages
        """
        
        full_texts = [p + c for p, c in zip(prompts, completions)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Reference log probs (frozen)
        with torch.no_grad():
            ref_log_probs_full = self.compute_log_probs_full_sequence(
                self.ref_model,
                inputs.input_ids,
                inputs.attention_mask,
                prompts
            )
        
        # Policy log probs (trainable)
        policy_log_probs_full = self.compute_log_probs_full_sequence(
            self.policy_model,
            inputs.input_ids,
            inputs.attention_mask,
            prompts,
            is_grad_enabled=True
        )
    
        kl_div = policy_log_probs_full - ref_log_probs_full
        
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        # ========================================================================
        # ADVANTAGE NORMALIZATION - Group level (same prompt)
        # ========================================================================
        if len(rewards_tensor) > 1:
            # Normalize advantages within the group
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        else:
            advantages = rewards_tensor
        
        # ========================================================================
        # GRPO LOSS: Policy gradient + KL penalty
        # ========================================================================
        log_ratio = policy_log_probs_full - ref_log_probs_full
        loss = -(advantages * log_ratio).mean() + self.kl_beta * log_ratio.mean()

        
        return loss

    def compute_log_probs_full_sequence(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompts: List[str], is_grad_enabled: bool = False) -> torch.Tensor:
        """
        Computes the sum of log probabilities for the completion tokens only.
        Used for both policy and reference models in the GRPO loss calculation.
        """
        
        if is_grad_enabled:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # Create a mask for completion tokens only
        encoded = self.tokenizer(prompts, add_special_tokens=False)
        prompt_lengths = [len(x) for x in encoded["input_ids"]]
        completion_mask = attention_mask[:, 1:].clone()
        
        for i, prompt_len in enumerate(prompt_lengths):
            completion_start = prompt_len - 1 # Adjusted for the shifted sequence (labels = input_ids[1:])
            # Mask out the prompt tokens
            completion_mask[i, :completion_start] = 0
            # Also mask out tokens beyond the completion length
            completion_end = min(completion_start + self.max_completion_length, completion_mask.shape[1])
            if completion_end < completion_mask.shape[1]:
                completion_mask[i, completion_end:] = 0
        
        token_log_probs = token_log_probs * completion_mask
        
        # Sum the log probabilities over the completion tokens for each example
        return token_log_probs.sum(dim=1)
    
    def train_step(self, batch: List[Dict], step: int):
        """
        Single training step with fixed loss handling
        """
        
        prompts = [self.format_prompt(ex) for ex in batch]
        
        total_loss = 0.0
        all_rewards = []
        num_loss_terms = 0
        
        self.policy_model.train()
        
        completions_per_prompt, _ = self.generate_completions(
            prompts, 
            self.policy_model, 
            num_completions=self.num_completions_per_prompt
        )
        
        rewards_per_prompt = []
        for i, example in enumerate(batch):
            completions_for_this_prompt = completions_per_prompt[i]
            rewards_for_this_prompt = self.compute_rewards(
                [example] * len(completions_for_this_prompt),
                completions_for_this_prompt
            )
            rewards_per_prompt.append(rewards_for_this_prompt)
            all_rewards.extend(rewards_for_this_prompt)
            
            for comp, rew in zip(completions_for_this_prompt, rewards_for_this_prompt):
                self.replay_buffer.append({
                    'prompt': prompts[i],
                    'completion': comp,
                    'reward': rew,
                    'example': example,
                    'step': step  # NUOVO
                })
        
        # NUOVO: Limita replay buffer e rimuovi sample vecchi
        if len(self.replay_buffer) > 2000:
            self.replay_buffer.sort(key=lambda x: x.get('step', 0), reverse=True)
            self.replay_buffer = self.replay_buffer[:2000]
        
        # ON-POLICY LOSS
        for i, prompt in enumerate(prompts):
            completions_group = completions_per_prompt[i]
            rewards_group = rewards_per_prompt[i]
            
            prompts_replicated = [prompt] * len(completions_group)
            
            loss = self.grpo_loss(
                prompts_replicated,
                completions_group,
                rewards_group,
                is_on_policy=True
            )
            
            loss.backward()
            total_loss += loss.item()
            num_loss_terms += 1
        
        # OFF-POLICY LOSS con filtro per sample recenti
        if len(self.replay_buffer) >= self.effective_batch_size:
            # NUOVO: Usa solo sample recenti
            recent_buffer = [s for s in self.replay_buffer 
                           if step - s.get('step', 0) < 500]
            
            if len(recent_buffer) >= self.effective_batch_size:
                samples_size = min(len(batch) * self.num_completions_per_prompt, 
                                 len(recent_buffer))
                samples = random.sample(recent_buffer, samples_size)
                
                samples_by_prompt = {}
                for s in samples:
                    prompt_key = s['prompt']
                    if prompt_key not in samples_by_prompt:
                        samples_by_prompt[prompt_key] = {'completions': [], 'rewards': []}
                    samples_by_prompt[prompt_key]['completions'].append(s['completion'])
                    samples_by_prompt[prompt_key]['rewards'].append(s['reward'])
                
                for prompt_key, data in samples_by_prompt.items():
                    prompts_replicated = [prompt_key] * len(data['completions'])
                    
                    loss_off = self.grpo_loss(
                        prompts_replicated,
                        data['completions'],
                        data['rewards'],
                        is_on_policy=False
                    )
                    
                    # NUOVO: Scala il loss off-policy
                    loss_off = loss_off * 0.5
                    loss_off.backward()
                    total_loss += loss_off.item()
                    num_loss_terms += 1
        
        # NUOVO: Media corretta del loss
        avg_loss = total_loss / max(num_loss_terms, 1)
        
        return avg_loss, all_rewards


def main():
    MODEL_NAME = "./models/GRPO_Hybrid_4B"
    TRAIN_DATA = "./data/BFCL/dataset/train.json"
    TEST_DATA = "./data/BFCL/dataset/test.json"
    OUTPUT_DIR = "./outputs/BFCL_Qwen3-4B"
    
    # Set to checkpoint path to resume training, or None to start fresh
    RESUME_FROM_CHECKPOINT = None

    trainer = GRPOTrainer(
        model_name=MODEL_NAME,
        train_data_path=TRAIN_DATA,
        test_data_path=TEST_DATA,
        output_dir=OUTPUT_DIR,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,#2
        num_completions_per_prompt=6,#4
        learning_rate=5e-7,
        max_grad_norm=0.3,#0.3
        kl_beta=0.001,
        ref_update_steps=500,#200
        max_completion_length=512,
        num_epochs=100,
        warmup_steps=50,
        mu=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        checkpoint_path=RESUME_FROM_CHECKPOINT 
    )

    # DEBUG: Test with first 3 examples
    print("\n" + "="*80)
    print("TESTING PROMPT FORMAT AND EVALUATION")
    print("="*80)
    
    examples_to_debug = trainer.train_dataset[:3]
    
    for i, example in enumerate(examples_to_debug):
        print(f"\n{'='*80}")
        print(f"Example {i+1}")
        print(f"{'='*80}")
        print(f"Query: {example['query']}")
        print(f"\nGround Truth ({len(example['ground_truth'])} functions required):")
        for gt in example['ground_truth']:
            print(f"  • {gt['name']}")
            if gt.get('parameters'):
                print(f"    Parameters: {gt['parameters']}")
        
        print(f"\nAvailable Functions:")
        for func in example['available_functions']:
            print(f"  • {func['name']}")
        
        prompt = trainer.format_prompt(example)
        print(f"\n--- FORMATTED PROMPT (first 800 chars) ---")
        print(prompt[:800])
        print("...")
        
        completion, _ = trainer.generate_completions([prompt], trainer.policy_model)
        print(f"\n--- MODEL COMPLETION ---")
        print(completion[0])
        
        # Test extraction
        extracted = trainer.evaluator.extract_function_calls(completion[0])
        print(f"\n--- EXTRACTED FUNCTIONS ({len(extracted)} found) ---")
        for func in extracted:
            print(f"  • {func['name']}")
            if func.get('parameters'):
                print(f"    Parameters: {func['parameters']}")
        
        # Check coverage
        gt_names = {gt['name'] for gt in example['ground_truth']}
        completion_names = {call['name'] for call in extracted}
        missing = gt_names - completion_names
        extra = completion_names - gt_names
        
        print(f"\n--- COVERAGE ANALYSIS ---")
        print(f"Required: {len(gt_names)}")
        print(f"Called: {len(completion_names)}")
        print(f"Correct: {len(gt_names.intersection(completion_names))}")
        if missing:
            print(f"Missing: {', '.join(missing)}")
        if extra:
            print(f"Extra: {', '.join(extra)}")
        
# Check format (Updated to handle list output and new tags)
        
        # Extract the completion string, handling the output as a list
        output_str = ""
        if isinstance(completion, list) and completion:
            # Make sure the first element is a string before using it
            output_str = str(completion[0]).lower()
        
        # Check for new tags
        has_reasoning = '<start_working_out>' in output_str and '</end_working_out>' in output_str
        has_solution = '<solution>' in output_str and '</solution>' in output_str
        
        print(f"\n--- FORMAT CHECK (Updated) ---")
        print(f"Has <start_working_out>/</end_working_out> tag: {has_reasoning}")
        print(f"Has <SOLUTION>/</SOLUTION> tag: {has_solution}")
        
        # ... il resto della funzione continua ...
        
        reward = trainer.compute_rewards([example], completion)
        print(f"\n--- REWARD ---")
        print(f"Reward: {reward[0]:.3f}")
        print()
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\nModel saved in: {trainer.output_dir}/final_model/")

if __name__ == "__main__":
    main()