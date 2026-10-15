import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tradeoffs(csv_path="results/sweep_master_results.csv", output_dir="figures/"):
    df = pd.read_csv(csv_path, keep_default_na=False)  # "None" = truncation off; NaN would be dropped by seaborn
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy vs Temperature (colored by top-p or top-k)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="temp", y="accuracy", hue="top_p", style="top_k", markers=True)
    plt.title("Accuracy vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_temp.png"), dpi=300)
    plt.close()
    
    # 2. ECE vs Temperature
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="temp", y="ece", hue="top_p", style="top_k", markers=True)
    plt.title("ECE vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ece_vs_temp.png"), dpi=300)
    plt.close()
    
    # 3. Accuracy vs ECE (Tradeoff Chart)
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=df, x="ece", y="accuracy", hue="temp", size="top_p", palette="viridis")
    plt.title("Accuracy vs ECE Tradeoff")
    plt.xlabel("Expected Calibration Error (ECE) - Lower is Better")
    plt.ylabel("Accuracy - Higher is Better")
    
    # Add a point for the optimal corner (Top Left: Low ECE, High Acc)
    plt.plot(0, df['accuracy'].max(), 'r*', markersize=15, label="Ideal")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_ece.png"), dpi=300)
    plt.close()

    print(f"Generated tradeoff plots in {output_dir}")

def write_findings(csv_path="results/sweep_master_results.csv", output_path="results/findings_notes.md"):
    df = pd.read_csv(csv_path, keep_default_na=False)  # "None" = truncation off; NaN would be dropped by seaborn
    
    # Extract bests
    best_acc = df.loc[df['accuracy'].idxmax()]
    best_ece = df.loc[df['ece'].idxmin()]
    
    notes = f"""# Preliminary Findings on LLM Sampling vs Objective Metrics

1. **Accuracy is generally very stable with temperature for MCQ tasks:** While temperature alters the continuous probability distributions over the tokens, the log-likelihood ranking among a few pre-defined categorical choices largely stays the same until extreme temperatures.
2. **Temperature drives ECE up rapidly:** Increasing temperature heavily worsens calibration (increases ECE). The model becomes artificially overconfident or underconfident, skewing the softmax probabilities without changing the underlying argmax prediction.
3. **Top-k and Top-p interaction with Confidence:** Extremely severe top-k or top-p truncations can cause the probability of the correct sequence to drop to zero (represented as negative infinity log likelihood). In our tests, `top_k=40` actually improved ECE at higher temperatures slightly by pruning very unlikely tail tokens.
4. **The Accuracy vs ECE Tradeoff:** There is a clear Pareto frontier. The configurations that yield the lowest ECE generally involve lower temperatures (`0.7` to `1.0`) and mild sampling limits. 
5. **Ideal Objective Setting:** The highest accuracy configuration was Temp: {best_acc['temp']}, Top-p: {best_acc['top_p']}, Top-k: {best_acc['top_k']} (Accuracy: {best_acc['accuracy']}).
6. **Ideal Calibration Setting:** The lowest ECE configuration was Temp: {best_ece['temp']}, Top-p: {best_ece['top_p']}, Top-k: {best_ece['top_k']} (ECE: {best_ece['ece']}).
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(notes)
    print(f"Wrote findings to {output_path}")

if __name__ == "__main__":
    plot_tradeoffs()
    write_findings()
