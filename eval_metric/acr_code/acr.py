
# Import Unsloth first
import unsloth
from unsloth import FastLanguageModel

# Other imports
import pandas as pd
import torch
from tqdm import tqdm
import logging
import sys
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def calculate_acr_score(target_length, num_tokens, success):
    """
    Calculate the raw Adversarial Compression Ratio (ACR) score.
    
    Args:
        target_length: Length of the target string in tokens
        num_tokens: Number of tokens in the prompt
        success: Whether the target was successfully generated
    
    Returns:
        Raw ACR score (float)
    """
    if not success:
        return 0.0
    
    # ACR is the ratio of target length to prompt length.
    acr = target_length / max(1, num_tokens)
    return acr

def prep_text(input_str, target_str, tokenizer, system_prompt, chat_template, num_free_tokens, device, dtype=None):
    """Modified prep_text to ensure consistent dtype"""
    # Get tokens
    input_tokens = tokenizer.encode(input_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    target_tokens = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    system_prompt_tokens = tokenizer.encode(system_prompt, return_tensors="pt", add_special_tokens=False).to(device=device)
    chat_template_tokens = (
        tokenizer.encode(chat_template[0], return_tensors="pt", add_special_tokens=False).to(device=device),
        tokenizer.encode(chat_template[1], return_tensors="pt", add_special_tokens=False).to(device=device)
    )
    
    # Generate random tokens for the free token section
    free_tokens = torch.randint(0, tokenizer.vocab_size, (1, num_free_tokens), device=device)
    
    # Concatenate all tokens
    input_ids = torch.cat(
        (chat_template_tokens[0], system_prompt_tokens, input_tokens, free_tokens, chat_template_tokens[1], target_tokens), 
        dim=1
    ).squeeze().long()  # Ensure long dtype for input_ids
    
    # Create slices
    tokens_before_free = chat_template_tokens[0].size(-1) + system_prompt_tokens.size(-1) + input_tokens.size(-1)
    free_token_slice = slice(tokens_before_free, tokens_before_free + free_tokens.size(-1))
    input_slice = slice(0, input_ids.size(-1) - target_tokens.size(-1))
    target_slice = slice(input_ids.size(-1) - target_tokens.size(-1), input_ids.size(-1))
    loss_slice = slice(input_ids.size(-1) - target_tokens.size(-1) - 1, input_ids.size(-1) - 1)
    
    return input_ids, free_token_slice, input_slice, target_slice, loss_slice

def check_output_with_hard_tokens(model, input_ids, target_slice, loss_slice):
    """Check if the model outputs match the target tokens"""
    with torch.no_grad():
        output = model(input_ids)
        match = (output.logits[0, loss_slice].argmax(-1) == input_ids[0, target_slice].squeeze()).all()
    return match

def random_search_optimize(model, input_ids, input_slice, free_token_slice, target_slice, loss_slice, 
                           num_steps=20, batch_size=32, mini_batch_size=8):
    """Random search optimization with consistent dtypes"""
    # Get model's dtype
    model_dtype = next(model.parameters()).dtype
    device = input_ids.device
    
    logger.info(f"Using model dtype: {model_dtype}")
    
    # Get vocab size
    vocab_size = model.config.vocab_size
    
    # Best solution tracking
    best_loss = float('inf')
    best_input = input_ids.clone()
    
    # Number of free tokens
    num_free_tokens = free_token_slice.stop - free_token_slice.start
    logger.info(f"Optimizing {num_free_tokens} free tokens")
    
    with torch.no_grad():  # No gradients needed for random search
        for step in range(num_steps):
            # Create candidates
            candidates = []
            for _ in range(batch_size):
                candidate = best_input.clone()
                
                # Modify 1-3 tokens
                num_to_change = min(3, num_free_tokens)
                positions = torch.randperm(num_free_tokens)[:num_to_change] + free_token_slice.start
                
                for pos in positions:
                    # Generate a random token
                    new_token = torch.randint(0, vocab_size, (1,), device=device).item()
                    candidate[pos] = new_token
                    
                candidates.append(candidate)
            
            # Convert to tensor
            candidates = torch.stack(candidates)
            
            # Evaluate candidates
            losses = []
            for i in range(0, batch_size, mini_batch_size):
                end = min(i + mini_batch_size, batch_size)
                try:
                    batch = candidates[i:end]
                    outputs = model(input_ids=batch)
                    
                    # Calculate losses
                    for j in range(len(batch)):
                        # Get logits and targets
                        logits = outputs.logits[j, loss_slice]
                        targets = batch[j, target_slice].squeeze()
                        
                        # Calculate loss
                        loss = torch.nn.functional.cross_entropy(logits, targets)
                        losses.append(loss.item())
                except Exception as e:
                    logger.error(f"Error in mini-batch {i}-{end}: {str(e)}")
                    # Assign high loss to failed candidates
                    losses.extend([float('inf')] * (end - i))
            
            # Find best candidate
            if losses:
                best_idx = min(range(len(losses)), key=lambda i: losses[i])
                current_loss = losses[best_idx]
                current_input = candidates[best_idx]
                
                # Check for match
                try:
                    outputs = model(input_ids=current_input.unsqueeze(0))
                    predicted = outputs.logits[0, loss_slice].argmax(dim=-1)
                    target = current_input[target_slice].squeeze()
                    match = (predicted == target).all().item()
                    
                    logger.info(f"Step {step+1} - Loss: {current_loss:.6f}, Match: {match}")
                    
                    # Update best if improved
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_input = current_input.clone()
                    
                    # Early stopping if match found
                    if match:
                        logger.info("Match found! Stopping search.")
                        best_input = current_input.clone()
                        break
                except Exception as e:
                    logger.error(f"Error checking match: {str(e)}")
            else:
                logger.warning(f"No valid candidates in step {step+1}")
    
    # Check final solution
    try:
        with torch.no_grad():
            outputs = model(input_ids=best_input.unsqueeze(0))
            match = (outputs.logits[0, loss_slice].argmax(-1) == best_input[target_slice].squeeze()).all().item()
    except Exception as e:
        logger.error(f"Error checking final solution: {str(e)}")
        match = False
    
    return {
        "input_ids": best_input,
        "success": match,
        "loss": best_loss
    }

def minimize_prompt(model, tokenizer, input_str, target_str, system_prompt, chat_template, device, 
                    max_tokens=300, num_steps=20, batch_size=32, mini_batch_size=8):
    """Find minimal prompt with consistent dtypes"""
    # Get model's dtype
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Model dtype: {model_dtype}")
    
    # Initial tokens
    n_tokens_in_prompt = 5
    running_max = max_tokens
    running_min = 0
    success = False
    best_prompt = None
    best_loss = float('inf')
    current_best_input = None  # Keep track of current best input even if not successful
    done = False
    best_slices = (None, None, None, None)
    best_n_tokens = None
    
    while not done:
        logger.info(f"Trying with {n_tokens_in_prompt} tokens in prompt")
        
        # Prepare text with consistent dtype
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prep_text(
            input_str, target_str, tokenizer, system_prompt, chat_template, 
            n_tokens_in_prompt, device, dtype=model_dtype
        )
        
        # Save this input_ids as a fallback
        if current_best_input is None:
            current_best_input = input_ids.clone()
        
        if running_max == -1:
            running_max = (target_slice.stop - target_slice.start) * 5
            
        # Run optimization
        solution = random_search_optimize(
            model, input_ids, input_slice, free_token_slice, target_slice, 
            loss_slice, num_steps=num_steps, batch_size=batch_size, mini_batch_size=mini_batch_size
        )
        
        # Update current best if this solution is better
        if solution.get("loss", float('inf')) < best_loss:
            best_loss = solution.get("loss", float('inf'))
            current_best_input = solution["input_ids"].clone()
        
        # Check if target is achieved
        target_acquired = solution.get("success", False)
        
        if target_acquired:
            logger.info(f"Target acquired with {n_tokens_in_prompt} tokens")
            running_max = n_tokens_in_prompt
            success = True
            best_prompt = solution["input_ids"].clone()
            best_n_tokens = n_tokens_in_prompt
            new_num_tokens = n_tokens_in_prompt - 1
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
        else:
            logger.info(f"Target NOT acquired with {n_tokens_in_prompt} tokens")
            # Linear growth - add 5 tokens each time
            new_num_tokens = n_tokens_in_prompt + 5
            running_min = n_tokens_in_prompt
            # Increase steps for next iteration
            num_steps = int(num_steps * 1.2)
            
        # Check if we're done
        if new_num_tokens >= running_max:
            logger.info(f"Reached maximum token count: {running_max}")
            done = True
        elif new_num_tokens <= running_min:
            logger.info(f"No more token counts to try")
            done = True
        else:
            n_tokens_in_prompt = new_num_tokens
            
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Use current_best_input as fallback if no successful prompt was found
    if best_prompt is None:
        logger.warning("No successful prompt found. Using best unsuccessful attempt.")
        best_prompt = current_best_input
        best_n_tokens = running_max  # Use maximum tokens tried as the number of tokens
    
    # Prepare output
    output = {
        "free_token_slice": best_slices[0] if best_slices[0] is not None else free_token_slice,
        "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
        "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
        "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
        "success": success,
        "num_free_tokens": best_n_tokens if best_n_tokens is not None else running_max,
        "input_ids": best_prompt,
        "loss": best_loss
    }
    
    return output

def main():
    # Configuration
    csv_path = '<filtered_gsmk8_.csv file path>'
    model_name = "sohamwasmatkar/lora_model_25"
    num_samples = 1
    
    # Load data
    logger.info(f"Loading data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Total rows: {len(df)}")
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return
        
    # Take sample
    sample_df = df.head(num_samples)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,  # Explicitly set dtype
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
        
    # Process samples
    results_list = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing samples"):
        try:
            logger.info(f"Processing sample {idx}")
            target_str = row['final_answer']
            logger.info(f"Target string: {target_str}")
            
            # Get optimal prompt
            solution = minimize_prompt(
                model, tokenizer, " ", target_str, "", ("", ""), device,
                max_tokens=40, num_steps=20, batch_size=16, mini_batch_size=4  # Reduced batch sizes
            )
            
            # Process results
            if solution["success"]:
                logger.info("Success! Target acquired.")
                
                input_ids = solution["input_ids"]
                input_slice = solution["input_slice"]
                target_slice = solution["target_slice"]
                num_tokens = solution["num_free_tokens"]
                
                # Calculate target length
                target_length = target_slice.stop - target_slice.start
                
                # Calculate raw ACR score
                raw_acr = calculate_acr_score(target_length, num_tokens, True)
                # Binary ACR flag: 1 for success, 0 for failure
                binary_acr = 1
                
                logger.info(f"Target length: {target_length}, Tokens used: {num_tokens}, Raw ACR: {raw_acr:.4f}, Success flag: {binary_acr}")
                
                # Decode optimal prompt
                optimal_prompt = tokenizer.decode(input_ids[input_slice], skip_special_tokens=True)
                logger.info(f"Optimal prompt: {optimal_prompt}")
                
                # Store results including both raw and binary ACR scores
                result = {
                    "target_str": target_str,
                    "optimal_prompt": optimal_prompt,
                    "success": True,
                    "target_length": target_length,
                    "num_tokens": num_tokens,
                    "acr_score": raw_acr,
                    "acr_binary": binary_acr,
                    "loss": solution["loss"]
                }
            else:
                logger.info("Failed to acquire target.")
                num_tokens = solution["num_free_tokens"]
                target_length = solution["target_slice"].stop - solution["target_slice"].start
                
                # For unsuccessful attempts, raw ACR score is 0 and binary flag is 0
                raw_acr = 0.0
                binary_acr = 0
                logger.info(f"Target length: {target_length}, Max tokens tried: {num_tokens}, Raw ACR: {raw_acr}, Success flag: {binary_acr}")
                
                result = {
                    "target_str": target_str,
                    "success": False,
                    "num_tokens": num_tokens,
                    "target_length": target_length,
                    "acr_score": raw_acr,
                    "acr_binary": binary_acr,
                    "loss": solution["loss"]
                }
                
            results_list.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            results_list.append({
                "target_str": target_str,
                "success": False,
                "error": str(e),
                "acr_score": 0.0,
                "acr_binary": 0
            })
            
    # Save results
    results_df = pd.DataFrame(results_list)
    output_path = 'acr_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Results summary:\n{results_df}")

if __name__ == "__main__":
    main()
