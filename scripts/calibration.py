import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import brier_score_loss, log_loss

def compute_ece(confidences, accuracies, num_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    bin_indices = np.digitize(confidences, bins) - 1
    # Handle the edge case where confidence is exactly 1.0 (goes to bin 10, but we want bin 9)
    bin_indices[bin_indices == num_bins] = num_bins - 1
    
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(confidences)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece

def plot_reliability_diagram(confidences, accuracies, num_bins=10, output_path="figures/reliability_diagram.png"):
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices[bin_indices == num_bins] = num_bins - 1
    
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_accs[bin_idx] = np.mean(accuracies[mask])
            bin_confs[bin_idx] = np.mean(confidences[mask])
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(bin_centers, bin_accs, width=1/num_bins, edgecolor="black", alpha=0.7, label="Outputs")
    
    # Gap visualization
    plt.bar(bin_centers, bin_confs - bin_accs, bottom=bin_accs, width=1/num_bins, 
            color="red", alpha=0.3, edgecolor="red", label="Gap")
    
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_selective_accuracy(confidences, accuracies, output_path="figures/selective_accuracy.png"):
    """Plots accuracy vs. confidence threshold (abstention)."""
    thresholds = np.linspace(0, 1, 100)
    accs = []
    coverages = []
    
    for t in thresholds:
        mask = confidences >= t
        coverage = np.mean(mask)
        if coverage > 0:
            acc = np.mean(accuracies[mask])
        else:
            acc = np.nan
        coverages.append(coverage)
        accs.append(acc)
        
    plt.figure(figsize=(8, 5))
    
    # Dual axis for coverage and accuracy
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(thresholds, accs, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.05])
    
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Coverage (Fraction of samples kept)', color=color)
    ax2.plot(thresholds, coverages, color=color, linestyle="--", linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1.05])
    
    plt.title("Selective Accuracy vs. Coverage")
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_calibration(csv_path="results/arc_easy_eval.csv"):
    df = pd.read_csv(csv_path)
    
    confidences = df['confidence'].values
    accuracies = df['is_correct'].values
    
    # Core Metrics
    accuracy = np.mean(accuracies)
    ece = compute_ece(confidences, accuracies)
    brier = brier_score_loss(accuracies, confidences)
    
    # Average NLL (if we use log loss. Note sklearn's log_loss expects probabilities for both classes, 
    # but here confidence is the predicted probability of the predicted class being correct)
    nll = log_loss(accuracies, confidences)
    
    print("-" * 30)
    print(f"Calibration Evaluation for {csv_path}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ECE:       {ece:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"NLL:       {nll:.4f}")
    print("-" * 30)
    
    plot_reliability_diagram(confidences, accuracies)
    plot_selective_accuracy(confidences, accuracies)
    print("Saved plots to figures/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/arc_easy_eval.csv")
    args = parser.parse_args()
    
    evaluate_calibration(args.input)
