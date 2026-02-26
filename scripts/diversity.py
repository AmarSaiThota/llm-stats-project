import os
import itertools
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')

def compute_distinct_n(texts, n=1):
    """Computes Distinct-1 or Distinct-2: ratio of unique n-grams to total n-grams."""
    total_ngrams = 0
    unique_ngrams = set()
    
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
            
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def compute_self_bleu(texts):
    """Computes Self-BLEU by treating each sentence as hypothesis and others as references."""
    if len(texts) < 2:
        return 0.0
        
    tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
    bleu_scores = []
    smoother = SmoothingFunction().method1
    
    for i, hyp in enumerate(tokenized_texts):
        refs = [ref for j, ref in enumerate(tokenized_texts) if i != j]
        score = sentence_bleu(refs, hyp, smoothing_function=smoother)
        bleu_scores.append(score)
        
    return np.mean(bleu_scores)

def compute_embedding_dispersion(texts, model):
    """Computes mean pairwise cosine distance of sentence embeddings."""
    if len(texts) < 2:
        return 0.0
        
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # 1 - cosine similarity
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    distances = 1.0 - similarity_matrix
    
    # Average upper triangular excluding diagonal
    num_samples = len(texts)
    upper_tri_indices = torch.triu_indices(num_samples, num_samples, offset=1)
    pairwise_distances = distances[upper_tri_indices[0], upper_tri_indices[1]]
    
    return pairwise_distances.mean().item()

def generate_and_evaluate_diversity(
    model_name="gpt2",
    prompts=["Once upon a time,", "The most important discovery of the 20th century was", "Artificial Intelligence is"],
    num_samples_per_prompt=5,
    max_new_tokens=40,
    device="cpu"
):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print("Loading SentenceTransformer for embeddings...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    temperatures = [0.2, 0.7, 1.0, 1.3]
    top_ps = [1.0, 0.9]
    top_ks = [0, 40]
    
    configs = list(itertools.product(temperatures, top_ps, top_ks))
    results = []
    
    for temp, top_p, top_k in tqdm(configs, desc="Diversity Sweep"):
        all_distinct_1 = []
        all_distinct_2 = []
        all_self_bleu = []
        all_dispersion = []
        
        for prompt in prompts:
            # Generate N texts
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            texts = []
            
            with torch.no_grad():
                for _ in range(num_samples_per_prompt):
                    # Random seed for variability
                    seed = torch.randint(0, 100000, (1,)).item()
                    set_seed(seed)
                    
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True if temp > 0 else False,
                        "pad_token_id": tokenizer.pad_token_id,
                        "temperature": temp
                    }
                    if top_p < 1.0: gen_kwargs["top_p"] = top_p
                    if top_k > 0: gen_kwargs["top_k"] = top_k
                    
                    try:
                        outputs = model.generate(**inputs, **gen_kwargs)
                        generated_sequence = outputs[0]
                        output_text = tokenizer.decode(generated_sequence[input_length:], skip_special_tokens=True).strip()
                        texts.append(output_text)
                    except Exception as e:
                        print(f"Failed generation for {temp}/{top_p}/{top_k}: {e}")
                        texts.append("")
                        
            # Filter empty texts
            texts = [t for t in texts if len(t.strip()) > 3]
            
            if len(texts) < 2:
                continue
                
            dist1 = compute_distinct_n(texts, 1)
            dist2 = compute_distinct_n(texts, 2)
            sb = compute_self_bleu(texts)
            disp = compute_embedding_dispersion(texts, embed_model)
            
            all_distinct_1.append(dist1)
            all_distinct_2.append(dist2)
            all_self_bleu.append(sb)
            all_dispersion.append(disp)
            
        if not all_distinct_1:
            continue
            
        results.append({
            "temp": temp,
            "top_p": top_p if top_p<1.0 else "None",
            "top_k": top_k if top_k>0 else "None",
            "distinct_1": np.mean(all_distinct_1),
            "distinct_2": np.mean(all_distinct_2),
            "self_bleu": np.mean(all_self_bleu),
            "embedding_dispersion": np.mean(all_dispersion)
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/diversity_metrics.csv", index=False)
    
    # Simple Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs("figures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.lineplot(data=df, x="temp", y="distinct_2", hue="top_p", style="top_k", ax=axes[0])
    axes[0].set_title("Distinct-2 vs Temp")
    
    sns.lineplot(data=df, x="temp", y="self_bleu", hue="top_p", style="top_k", ax=axes[1])
    axes[1].set_title("Self-BLEU vs Temp (Lower=More Diverse)")
    
    sns.lineplot(data=df, x="temp", y="embedding_dispersion", hue="top_p", style="top_k", ax=axes[2])
    axes[2].set_title("Embedding Dispersion vs Temp")
    
    plt.tight_layout()
    plt.savefig("figures/diversity_plots.png", dpi=300)
    plt.close()
    
    print("Diversity experiments complete. Results in results/diversity_metrics.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    generate_and_evaluate_diversity(model_name=args.model, num_samples_per_prompt=args.samples, device=args.device)
