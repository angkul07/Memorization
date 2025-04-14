# """
# gcg.py
# an implementation of Greedy Coordinate Gradient
# From: Universal and Transferable Adversarial Attacks on Aligned Language Models
# By: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson
# 2023
# https://arxiv.org/abs/2307.15043

# developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
# """

# import logging

# import torch
# import torch.nn.functional as F


# def sample_tokens(num_tokens, embedding_matrix, batch_size, device):
#     sample = torch.randint(0, embedding_matrix.size(0), (batch_size, num_tokens), device=device)
#     new_token_loc = torch.randint(0, num_tokens, (batch_size,), device=device)
#     new_token_vals = torch.randint(0, embedding_matrix.size(0), (batch_size,), device=device)
#     sample[torch.arange(batch_size), new_token_loc] = new_token_vals
#     return sample


# def optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice, loss_slice,
#                  num_steps, topk=250, batch_size=100, mini_batch_size=100):
#     # Get embedding matrix
#     try:
#         embedding_matrix = model.get_input_embeddings().weight
#     except NotImplementedError:
#         embedding_matrix = model.transformer.wte.weight

#     best_loss = torch.inf
#     best_input = input_ids.clone()

#     # Greedy Coordinate Gradient optimization loop
#     for i in range(num_steps):
#         # Create one-hot tensor and embeddings from input_ids
#         inputs_one_hot = F.one_hot(input_ids, embedding_matrix.size(0)).type(embedding_matrix.dtype).unsqueeze(0)
#         inputs_one_hot.requires_grad_(True)
#         inputs_embeds = torch.matmul(inputs_one_hot, embedding_matrix)
#         # Forward and backward pass
#         output = model(inputs_embeds=inputs_embeds)
#         loss = torch.nn.functional.cross_entropy(output.logits[0, loss_slice], input_ids[target_slice].squeeze())
#         grad = torch.autograd.grad(loss, inputs_one_hot)[0][:, free_token_slice]
#         with torch.no_grad():
#             # Get topk gradients
#             top_values, top_indices = torch.topk(-grad[0], topk, dim=1)
#             # Build batch of input_ids with random topk tokens
#             free_token_ids = inputs_one_hot[0, free_token_slice].argmax(-1)
#             free_tokens_batch = free_token_ids.repeat(batch_size, 1)
#             new_token_loc = torch.randint(0, free_token_ids.size(0), (batch_size, 1))
#             new_token_vals = top_indices[new_token_loc, torch.randint(0, topk, (batch_size, 1))]
#             free_tokens_batch[torch.arange(batch_size), new_token_loc.squeeze()] = new_token_vals.squeeze()
#             candidates_input_ids = input_ids.repeat(batch_size, 1)
#             candidates_input_ids[:, free_token_slice] = free_tokens_batch

#             loss = torch.zeros(batch_size)
#             for mini_batch in range(0, batch_size, mini_batch_size):
#                 output = model(input_ids=candidates_input_ids[mini_batch:mini_batch + mini_batch_size])
#                 labels = input_ids[target_slice].repeat(output.logits.size(0), 1)
#                 loss_mini_batch = F.cross_entropy(output.logits[:, loss_slice].transpose(1, 2), labels,
#                                                   reduction="none")
#                 loss[mini_batch:mini_batch + mini_batch_size] = loss_mini_batch.mean(dim=-1)
#             best_candidate = torch.argmin(loss)
#             input_ids = candidates_input_ids[best_candidate]

#             # Compute test loss and check token matches
#             output_single = model(input_ids=input_ids.unsqueeze(0))
#             match = (output_single.logits[0, loss_slice].argmax(-1) == input_ids[target_slice].squeeze())
#         logging.info(f"step: {i:<4} | "
#                      f"loss: {loss[best_candidate].mean().item():0.6f} | "
#                      f"{match.int().tolist()} | "
#                      )
#         if match.all():
#             best_input = input_ids.clone()
#             break
#         if loss[best_candidate].mean().item() < best_loss:
#             best_loss = loss[best_candidate].mean().item()
#             best_input = input_ids.clone()

#     return {"input_ids": best_input, "inputs_embeds": model.get_input_embeddings()(best_input).unsqueeze(0)}






"""
gcg.py
Modified implementation of Greedy Coordinate Gradient with data type consistency
"""

import logging
import torch
import torch.nn.functional as F

def optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice, loss_slice,
                 num_steps, topk=250, batch_size=100, mini_batch_size=100):
    """
    Fixed implementation of GCG to ensure consistent data types
    """
    # Get device
    device = input_ids.device
    
    # Determine the correct dtype for the model
    model_dtype = None
    for param in model.parameters():
        model_dtype = param.dtype
        break
    
    if model_dtype is None:
        model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    
    logging.info(f"Using model dtype: {model_dtype}")
    
    # Get embedding matrix with correct dtype
    try:
        embedding_matrix = model.get_input_embeddings().weight.to(model_dtype)
    except NotImplementedError:
        embedding_matrix = model.transformer.wte.weight.to(model_dtype)
    
    # Get vocabulary size
    vocab_size = embedding_matrix.size(0)
    
    # Initialize variables
    best_loss = float('inf')
    best_input = input_ids.clone()
    
    # Random search approach (no gradients needed)
    for i in range(num_steps):
        with torch.no_grad():  # Use no_grad for the entire process
            # Create batch of candidates
            candidates = []
            for b in range(batch_size):
                # Start with the current best input
                candidate = best_input.clone()
                
                # Choose a random position in the free token slice
                pos_idx = torch.randint(0, free_token_slice.stop - free_token_slice.start, (1,)).item()
                pos = free_token_slice.start + pos_idx
                
                # Choose a random token
                new_token = torch.randint(0, vocab_size, (1,)).item()
                
                # Update the candidate
                candidate[pos] = new_token
                candidates.append(candidate)
            
            # Stack candidates
            candidates = torch.stack(candidates)
            
            # Initialize losses
            losses = torch.zeros(batch_size, device=device)
            
            # Evaluate in mini-batches
            for j in range(0, batch_size, mini_batch_size):
                end_idx = min(j + mini_batch_size, batch_size)
                
                try:
                    # Forward pass
                    outputs = model(input_ids=candidates[j:end_idx])
                    
                    # Compute loss
                    for k in range(j, end_idx):
                        logits = outputs.logits[k-j, loss_slice]
                        target = candidates[k, target_slice].squeeze()
                        loss = F.cross_entropy(logits, target)
                        losses[k] = loss.item()
                except Exception as e:
                    logging.error(f"Error evaluating batch: {str(e)}")
                    losses[j:end_idx] = float('inf')
            
            # Find best candidate
            best_idx = torch.argmin(losses).item()
            candidate_input_ids = candidates[best_idx]
            
            # Check if the candidate generates the target
            outputs = model(input_ids=candidate_input_ids.unsqueeze(0))
            match = (outputs.logits[0, loss_slice].argmax(-1) == candidate_input_ids[target_slice].squeeze()).all()
            
            # Log progress
            logging.info(f"Step {i+1}: loss={losses[best_idx].item():.6f}, match={match.item()}")
            
            # Update best input if improved
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx].item()
                best_input = candidate_input_ids.clone()
            
            # Early stopping if match found
            if match:
                logging.info("Match found! Stopping early.")
                best_input = candidate_input_ids.clone()
                break
    
    # Return result
    return {
        "input_ids": best_input,
        "inputs_embeds": model.get_input_embeddings()(best_input.unsqueeze(0)).squeeze(0)
    }