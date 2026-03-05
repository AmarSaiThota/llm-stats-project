# Preliminary Findings on LLM Sampling vs Objective Metrics

1. **Accuracy is generally very stable with temperature for MCQ tasks:** While temperature alters the continuous probability distributions over the tokens, the log-likelihood ranking among a few pre-defined categorical choices largely stays the same until extreme temperatures.
2. **Temperature drives ECE up rapidly:** Increasing temperature heavily worsens calibration (increases ECE). The model becomes artificially overconfident or underconfident, skewing the softmax probabilities without changing the underlying argmax prediction.
3. **Top-k and Top-p interaction with Confidence:** Extremely severe top-k or top-p truncations can cause the probability of the correct sequence to drop to zero (represented as negative infinity log likelihood). In our tests, `top_k=40` actually improved ECE at higher temperatures slightly by pruning very unlikely tail tokens.
4. **The Accuracy vs ECE Tradeoff:** There is a clear Pareto frontier. The configurations that yield the lowest ECE generally involve lower temperatures (`0.7` to `1.0`) and mild sampling limits. 
5. **Ideal Objective Setting:** The highest accuracy configuration was Temp: 0.2, Top-p: nan, Top-k: 40.0 (Accuracy: 0.44).
6. **Ideal Calibration Setting:** The lowest ECE configuration was Temp: 0.7, Top-p: 0.9, Top-k: 100.0 (ECE: 0.1219999999999999).
