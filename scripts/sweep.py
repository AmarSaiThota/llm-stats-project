import argparse
import itertools
import os
import pandas as pd
from eval_mcq import evaluate_arc
from calibration import evaluate_calibration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    # Grid definition
    temperatures = [0.2, 0.7, 1.0, 1.3]
    top_ps = [1.0, 0.9, 0.95]
    top_ks = [0, 40, 100]
    
    configs = list(itertools.product(temperatures, top_ps, top_ks))
    
    master_results = []
    
    results_dir = "results/sweep"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting sweep over {len(configs)} configurations...")
    
    for i, (temp, top_p, top_k) in enumerate(configs):
        print(f"\n--- Config {i+1}/{len(configs)}: temp={temp}, top_p={top_p}, top_k={top_k} ---")
        
        file_name = f"arc_T{temp}_P{top_p}_K{top_k}.csv"
        csv_path = os.path.join(results_dir, file_name)
        
        # Run evaluation to get predictions
        accuracy = evaluate_arc(
            model_name=args.model,
            split=args.split,
            num_samples=args.num_samples,
            output_csv=csv_path,
            device=args.device,
            temp=temp,
            top_p=top_p,
            top_k=top_k
        )
        
        # Load CSV and run calibration
        df = pd.read_csv(csv_path)
        confidences = df['confidence'].values
        accuracies = df['is_correct'].values
        
        from calibration import compute_ece
        from sklearn.metrics import brier_score_loss, log_loss
        
        ece = compute_ece(confidences, accuracies)
        brier = brier_score_loss(accuracies, confidences)
        nll = log_loss(accuracies, confidences)
        
        master_results.append({
            "temp": temp,
            "top_p": top_p if top_p < 1.0 else "None",
            "top_k": top_k if top_k > 0 else "None",
            "accuracy": accuracy,
            "ece": ece,
            "brier": brier,
            "nll": nll
        })
        
        print(f"Done -> Acc: {accuracy:.4f} | ECE: {ece:.4f} | Brier: {brier:.4f}")
        
    master_df = pd.DataFrame(master_results)
    master_csv = "results/sweep_master_results.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\nSweep complete! Saved to {master_csv}")

if __name__ == "__main__":
    main()
