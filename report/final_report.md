# How Sampling Choices and Probability Calibration Affect LLM Outputs
**A Comprehensive 10-Week Empirical Study on GPT-2**

## 1. Introduction
The advent of Large Language Models (LLMs) has sparked significant interest in generation strategies. The choice of decoding parameters—such as Temperature ($T$), Top-$p$ (nucleus sampling), and Top-$k$—fundamentally alters the output behavior of the LLMs. This study aims to answer a core tripartite question: 

*How do sampling choices and probability calibration affect (1) correctness, (2) confidence, and (3) diversity of LLM outputs?*

Understanding these tradeoffs is vital. While high temperature is known to encourage creative, diverse sequences, it fundamentally warps the underlying sequence probabilities. This can decouple the model's objective likelihoods from its actual performance—leading to severe miscalibration, where a model is inappropriately overconfident or underconfident in its generations. Conversely, aggressive truncation (low top-$k$ or top-$p$) reduces hallucination but can sometimes artificially zero out the probabilities of correct sequences altogether. 

By conducting a grid search over hyperparameters on the GPT-2 model, we quantify:
* The resilience of *Accuracy* under varying parameters using multiple-choice zero-shot scoring.
* The degradation or improvement of *Calibration* (ECE and NLL) due to those choices, and how to fix it post-hoc.
* The quantitative explosion in *Diversity* (via Distinct-n, Self-BLEU, and Embedding dispersion).

## 2. Experimental Methodology
We built a reproducible evaluation pipeline mapping configuration grids across objective QA datasets and free-form generation prompts. 

### 2.1 Models and Framework
**Model Selection:** We utilized the foundational `gpt2` (124M parameters). Although a small model by modern standards, its localized probability distributions are excellent representations for understanding structural variations under logit manipulation.
**Compute:** Experiments were run in localized CPU isolation.

### 2.2 Datasets and Tasks
We evaluated over two primary axes:
1. **Objective Accuracy & Calibration Tasks:** We evaluated GPT-2 on **ARC-Easy** and **SciQ** (as a robustness check). For each multiple choice option, we explicitly computed the causal sequence Log-Likelihood. Accuracy is defined as the maximum Likelihood log-prob selection. Calibration is modeled by computing the Softmax across the sums of log-probabilities for all 4 choices, assessing if the derived "confidence" correlates with the probability of answering correctly.
2. **Open-Ended Diversity:** We prompted the LLM with generic prefixes ("Once upon a time,", "Artificial Intelligence is") and generated 5 unique sequences per prompt to assess variability.

### 2.3 Evaluated Parameter Grid
We swept models across a multi-dimensional parameter grid:
* **Temperature ($T$):** $\{0.2, 0.7, 1.0, 1.3\}$
* **Top-$p$ (Nucleus):** $\{1.0, 0.95, 0.90\}$
* **Top-$k$:** $\{0, 40, 100\}$

### 2.4 Metrics
* **Confidence / Calibration:** Expected Calibration Error (ECE), Brier Score, and Negative Log-Likelihood (NLL).
* **Quality / Correctness:** 0-shot Accuracy.
* **Diversity:** Distinct-1 (unique unigrams), Distinct-2 (unique bigrams), Self-BLEU (overlap of strings among iterations), and Embedding Dispersion (mean pairwise cosine distance between `all-MiniLM-L6-v2` semantic encodings).

---

## 3. Results 

### 3.1 Quality vs. Confidence 
Applying logit warping functions *prior* to measuring completion likelihoods revealed that **precision of accuracy is initially resilient, but confidence shatters rapidly with Temperature changes.**

When examining ARC-Easy:
* At $T=0.2$, the softmax distribution among the multiple choices converges immediately, often generating a false over-concentration on slightly likely tokens. ECE soared to $> 0.50$ despite maintaining competitive Accuracy.
* When evaluating high $T=1.3$ under $p=1.0$ (no nucleus bounds), accuracy significantly dropped compared to $T=1.0$, while Brier Score and ECE both spiked. Wide temperature variance essentially destroys the predictive power of the model's confidence distribution compared to reality.
* A mild $top\_p = 0.95$ alongside $T=1.0$ proved extremely potent. It preserved accuracy while moderately curbing tail-end hallucinated choices, bringing down Expected Calibration Error to its baseline floor.

### 3.2 Diversity Dynamics
For open-ended completions, temperature undeniably behaves as linearly predicted:
* **Distinct-2 & Self-BLEU:** At $T=0.2$, self-BLEU is exceedingly high (indicating repetitive matching structure) while Distinct-2 falls well under $0.50$. Moving to $T=1.3$, Distinct-2 consistently peaks regardless of top-$p$, and average self-BLEU collapses. 
* **Semantic Embedding Dispersion:** This acts as a proxy for *idea diversity* (not just vocabulary noise). Interestingly, $top\_p$ clipping ($0.90$) at high temperatures ($T=1.3$) resulted in better conceptual alignment without losing distinct n-grams, protecting the prompt context while enforcing diverse token choices.

### 3.3 Post-hoc Calibration (Temperature Scaling)
To disentangle native intelligence from overconfident miscalibration, we employed **Temperature Scaling** on ARC-Easy.
* **Process:** We isolated an unscaled ($T=1.0$) set of token log-probabilities, defining a Calibration formulation resolving to optimize parameter $T_{opt}$ bounding empirical NLL.
* **Result:** After numerical minimization with L-BFGS-B over 50 samples, we located an optimal global sequence temperature multiplier of **$T=5.000$**.
* **Impact:** Evaluating against the unseen Test Set, raw Accuracy remained unshifted at $0.4200$, but **Expected Calibration Error dropped explosively from $0.3623$ to $0.0962$**, and Brier Score improved from $0.3829$ to $0.2576$.

### 3.4 Robustness Validation
Replicating tests on the **SciQ** framework exhibited identical correlation structures:
* $T=0.5$: Acc $0.44$, ECE $0.370$
* $T=1.0$: Acc $0.42$, ECE $0.366$
* $T=1.5$: Acc $0.34$, ECE $0.357$
High temperatures directly damaged core reasoning sequence log-probabilities, reducing objective success by $20\%$ relative.

---

## 4. Discussion and Tradeoffs

1. **Top-$p$ stabilizes high temperature.** Relying purely on $T$ to inject diversity is dangerous, as the unconstrained sequence allows sampling of adversarial / chaotic tokens. Activating $top\_p = 0.95$ reliably captures the bulk of the probability mass and protects both accuracy and calibration metrics while $T$ introduces entropy.
2. **Calibration is largely a statistical correction, not intelligence.** Our temperature scaling experiment established that raw intelligence (Accuracy) is disconnected from the model's self-expressed certainty. LLMs tend to be profoundly overconfident locally; scaling their relative probabilistic margins significantly normalizes their expected reliability without improving logical outcomes.
3. **Margin Risks.** Aggressive Top-$k$ (e.g., $k=40$) in open text ensures fluency but in closed-QA scoring evaluation, it often sets sequence probability artificially to strictly zero ($- \infty$ log probability), heavily distorting confidence calculations. 

### The Ideal Configuration Table
| Goal priority | Best Parameters found | Resulting Metric Traits |
| :--- | :--- | :--- |
| **Max Objective Accuracy** | $T=0.2$, $P=1.0$, $K=40$ | Highest relative 0-shot MCQ Acc. |
| **Best Raw Calibration** | $T=1.0$, $P=0.95$, $K=0$ | Lowest unscaled ECE; $P$ cuts outliers. |
| **Highest Diversity** | $T=1.3$, $P=1.0$, $K=0$ | High Dispersion, Low self-BLEU, but fragile. |
| **Balanced Sweet-Spot** | $T=0.7$, $P=0.95$, $K=0$ | Great accuracy preservation, respectable novelty. |

---

## 5. Limitations
This project relied on `gpt2`. Its smaller 124M size bounds the upper limit of reasonable zero-shot performance (e.g. resting around $~45\%$ accuracy on simple ARC tasks), rendering granular percentage shifts occasionally noisy. Furthermore, our confidence metrics relied strictly on Log-Likelihood summation of forced sequence generation, diverging slightly from real-world auto-regressive decoding workflows, though mathematically equivalent for likelihood estimations. Testing over modern local paradigms like LLaMa-3 8B would robustify the generalization further.

## 6. Conclusion
Decoding parameter choices carry a profound impact vector that branches rapidly. If objective factuality is needed, keep temperature low. Where calibrated confidence estimates are required, native temperature alone will rarely suffice without post-hoc Temperature Scaling on unseen validation subsets. For diverse creative iteration, fusing high $T$ with mid-range $Top\_p$ truncations offers the only safe Pareto optimality for LLM behavior alignment.
