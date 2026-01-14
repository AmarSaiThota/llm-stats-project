import pandas as pd
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def get_choice_logprob(model, tokenizer, prompt, choice, device, temp=1.0, top_p=1.0, top_k=0):
    prompt_tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    choice_tokens = tokenizer(" " + choice, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    input_ids = torch.cat([prompt_tokens, choice_tokens]).unsqueeze(0).to(device)
    num_choice_tokens = len(choice_tokens)
    if num_choice_tokens == 0:
        return -1000.0, -1000.0
        
    with torch.no_grad():
        outputs = model(input_ids)

    # Get logits for the choice tokens
    # the logit at index i predicts token at index i+1
    start_idx = len(prompt_tokens) - 1
    end_idx = start_idx + num_choice_tokens
    
    logits = outputs.logits[0, start_idx:end_idx, :].clone() # Shape: (num_choice_tokens, vocab_size)
    
    # Apply warping
    if temp != 1.0 and temp > 0:
        logits = logits / temp
    
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    # Compute log probs
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Gather logs probs for the actual tokens
    actual_tokens = choice_tokens.to(device).unsqueeze(-1) # shape: (num_choice_tokens, 1)
    token_log_probs = log_probs.gather(1, actual_tokens).squeeze(-1)
    
    sum_logprob = token_log_probs.sum().item()
    avg_logprob = token_log_probs.mean().item()
    
    return sum_logprob, avg_logprob

def evaluate_arc(model_name="gpt2", split="validation", num_samples=None, output_csv="results/arc_easy_eval.csv", device="cpu", temp=1.0, top_p=1.0, top_k=0):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
    results = []
    
    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']

        best_score = -float('inf')
        predicted_label = choices['label'][0]
        
        choice_scores = {}
        prompt = f"Question: {question}\nAnswer:"
        
        for label, text in zip(choices['label'], choices['text']):
            sum_lp, avg_lp = get_choice_logprob(model, tokenizer, prompt, text, device, temp=temp, top_p=top_p, top_k=top_k)
            choice_scores[label] = {
                "sum_logprob": sum_lp,
                "avg_logprob": avg_lp,
                "text": text
            }
            if sum_lp > best_score:
                best_score = sum_lp
                predicted_label = label
                
        is_correct = (predicted_label == answer_key)
        
        sorted_scores = sorted([c['sum_logprob'] for c in choice_scores.values()], reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        all_sum_lps = list(c['sum_logprob'] for c in choice_scores.values())
        max_lp = max(all_sum_lps)
        if max_lp == -float('inf'):
            probs = [1.0 / len(all_sum_lps)] * len(all_sum_lps)
        else:
            exp_lps = [np.exp(lp - max_lp) for lp in all_sum_lps]
            sum_exp_lps = sum(exp_lps)
            probs = [exp_lp / sum_exp_lps for exp_lp in exp_lps]
        
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]

        top1_avg_logprob = choice_scores[predicted_label]['avg_logprob']
        
        results.append({
            "id": item["id"],
            "question": question,
            "answer_key": answer_key,
            "predicted_label": predicted_label,
            "is_correct": int(is_correct),
            "confidence": confidence,
            "top1_sum_logprob": best_score,
            "top1_avg_logprob": top1_avg_logprob,
            "margin": margin,
            "num_choices": len(choices['label'])
        })
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    accuracy = df["is_correct"].mean()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/arc_easy_eval.csv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    
    args = parser.parse_args()
    
    evaluate_arc(
        model_name=args.model,
        split=args.split,
        num_samples=args.num_samples,
        output_csv=args.output,
        device=args.device,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k
    )
