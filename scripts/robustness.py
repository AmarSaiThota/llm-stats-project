import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_mcq import get_choice_logprob

def evaluate_sciq(model_name="gpt2", num_samples=100, temps=[0.5, 1.0, 1.5], device="cpu"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print("Loading SciQ dataset...")
    dataset = load_dataset("sciq", split="validation")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    master_results = []
    
    for temp in temps:
        print(f"\n--- Testing SciQ at Temperature={temp} ---")
        results = []
        
        for item in tqdm(dataset):
            question = item['question']
            correct = item['correct_answer']
            d1 = item['distractor1']
            d2 = item['distractor2']
            d3 = item['distractor3']
            
            choices = [correct, d1, d2, d3]
            
            prompt = f"Question: {question}\nAnswer:"
            
            lps = []
            for c in choices:
                sum_lp, _ = get_choice_logprob(model, tokenizer, prompt, c, device, temp=temp)
                lps.append(sum_lp)
            
            best_score = max(lps)
            predicted_idx = lps.index(best_score)
            
            is_correct = (predicted_idx == 0) # correct is index 0
            
            max_lp = max(lps)
            if max_lp == -float('inf'):
                probs = [0.25, 0.25, 0.25, 0.25]
            else:
                exp_lps = [np.exp(lp - max_lp) for lp in lps]
                probs = [e / sum(exp_lps) for e in exp_lps]
                
            confidence = probs[predicted_idx]
            
            results.append({
                "is_correct": int(is_correct),
                "confidence": confidence
            })
            
        df = pd.DataFrame(results)
        
        from calibration import compute_ece
        from sklearn.metrics import brier_score_loss
        
        acc = df["is_correct"].mean()
        ece = compute_ece(df["confidence"].values, df["is_correct"].values)
        brier = brier_score_loss(df["is_correct"].values, df["confidence"].values)
        
        print(f"SciQ @ T={temp} -> Acc: {acc:.4f} | ECE: {ece:.4f} | Brier: {brier:.4f}")
        
        master_results.append({
            "temp": temp,
            "accuracy": acc,
            "ece": ece,
            "brier": brier
        })
        
    master_df = pd.DataFrame(master_results)
    os.makedirs("results", exist_ok=True)
    master_df.to_csv("results/robustness_sciq.csv", index=False)
    print("\nRobustness checks complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    evaluate_sciq(model_name=args.model, num_samples=args.samples, device=args.device)
