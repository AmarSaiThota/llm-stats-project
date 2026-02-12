import argparse
import itertools
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from calibration import compute_ece
from sklearn.metrics import brier_score_loss, log_loss

# Re-use our eval function to get the raw unscaled scores (Temp=1)
from eval_mcq import AutoModelForCausalLM, AutoTokenizer, load_dataset, tqdm, get_choice_logprob

def get_raw_scores(model, tokenizer, dataset, device):
    """Evaluates the dataset and returns raw sum_logprobs for all choices"""
    all_scores = []
    labels = []
    
    for item in tqdm(dataset, desc="Extracting raw scores"):
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']
        prompt = f"Question: {question}\nAnswer:"
        
        choice_scores = []
        is_correct_idx = 0
        
        for idx, (label, text) in enumerate(zip(choices['label'], choices['text'])):
            if label == answer_key:
                is_correct_idx = idx
            sum_lp, _ = get_choice_logprob(model, tokenizer, prompt, text, device, temp=1.0, top_p=1.0, top_k=0)
            choice_scores.append(sum_lp)
            
        all_scores.append(choice_scores)
        labels.append(is_correct_idx)
        
    return all_scores, labels

def evaluate_with_temp(raw_scores, labels, temp=1.0):
    """Apply temperature $T$, compute softmax and metrics."""
    confidences = []
    accuracies = []
    
    for scores, label_idx in zip(raw_scores, labels):
        # scale
        scaled_scores = [s / temp for s in scores]
        
        max_s = max(scaled_scores)
        if max_s == -float('inf'):
            probs = [1.0 / len(scaled_scores)] * len(scaled_scores)
        else:
            exp_s = [np.exp(s - max_s) for s in scaled_scores]
            sum_exp = sum(exp_s)
            probs = [e / sum_exp for e in exp_s]
            
        predicted_idx = np.argmax(probs)
        confidences.append(probs[predicted_idx])
        accuracies.append(1 if predicted_idx == label_idx else 0)
        
    ece = compute_ece(np.array(confidences), np.array(accuracies))
    brier = brier_score_loss(accuracies, confidences)
    acc = np.mean(accuracies)
    
    return acc, ece, brier, confidences

def nll_objective(T, raw_scores, labels):
    """Objective function for optimization: NLL with temperature T."""
    temp = T[0]
    nll = 0.0
    for scores, label_idx in zip(raw_scores, labels):
        scaled_scores = [s / temp for s in scores]
        max_s = max(scaled_scores)
        if max_s == -float('inf'):
            continue
            
        exp_s = [np.exp(s - max_s) for s in scaled_scores]
        sum_exp = sum(exp_s)
        probs = [e / sum_exp for e in exp_s]
        
        # NLL is -log probability of the TRUE label
        prob_true = probs[label_idx]
        # Avoid log(0)
        prob_true = max(prob_true, 1e-12)
        nll += -np.log(prob_true)
        
    return nll / len(raw_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--calib_samples", type=int, default=50)
    parser.add_argument("--test_samples", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()
    
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    # Split into calib and test
    calib_set = dataset.select(range(args.calib_samples))
    test_set = dataset.select(range(args.calib_samples, args.calib_samples + args.test_samples))
    
    print("\n--- Processing Calibration Set ---")
    calib_raw, calib_labels = get_raw_scores(model, tokenizer, calib_set, args.device)
    
    print("\n--- Processing Test Set ---")
    test_raw, test_labels = get_raw_scores(model, tokenizer, test_set, args.device)
    
    # 1. Base evaluation (T=1.0) on Test Set
    baseline_acc, baseline_ece, baseline_brier, _ = evaluate_with_temp(test_raw, test_labels, temp=1.0)
    
    # 2. Fit temperature on Calibration Set (minimize NLL)
    print("\nFitting Temperature on Calibration Set...")
    opt_result = minimize(
        nll_objective, 
        x0=[1.0], 
        args=(calib_raw, calib_labels),
        bounds=[(0.1, 5.0)],
        method='L-BFGS-B'
    )
    opt_T = opt_result.x[0]
    print(f"Optimal Temperature found: {opt_T:.3f}")
    
    # 3. Post-scaling evaluation on Test Set
    scaled_acc, scaled_ece, scaled_brier, _ = evaluate_with_temp(test_raw, test_labels, temp=opt_T)
    
    print("\n=== Post-hoc Temperature Scaling Results (Test Set) ===")
    print(f"Baseline (T=1.0):")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  ECE:      {baseline_ece:.4f}")
    print(f"  Brier:    {baseline_brier:.4f}")
    
    print(f"\nScaled (T={opt_T:.3f}):")
    print(f"  Accuracy: {scaled_acc:.4f}")
    print(f"  ECE:      {scaled_ece:.4f}")
    print(f"  Brier:    {scaled_brier:.4f}")
    
    # Save the before/after metrics
    results_df = pd.DataFrame([{
        "Metric": "Accuracy",
        "Baseline": baseline_acc,
        "Scaled": scaled_acc
    },
    {
        "Metric": "ECE",
        "Baseline": baseline_ece,
        "Scaled": scaled_ece
    },
    {
        "Metric": "Brier",
        "Baseline": baseline_brier,
        "Scaled": scaled_brier
    }])
    
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/temperature_scaling_results.csv", index=False)
    
    # (Optional) We could plot reliability diagram before and after 
    # but printing metrics satisfies the primary goal for Day 7.

if __name__ == "__main__":
    main()
