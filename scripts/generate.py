import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def compute_entropy(logits):
    """Computes entropy of the probability distribution from logits."""
    return torch.distributions.Categorical(logits=logits).entropy()

def generate_samples(
    model_name="gpt2",
    prompts=["The capital of France is"],
    output_file="results/baseline.jsonl",
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    typical_p=1.0,
    repetition_penalty=1.0,
    max_new_tokens=50,
    seed=42,
    device="cpu"
):
    set_seed(seed)
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f, torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # Setup generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
                "do_sample": True if temperature > 0 else False,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id
            }
            
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                if top_p < 1.0:
                    gen_kwargs["top_p"] = top_p
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k
                if typical_p < 1.0:
                    gen_kwargs["typical_p"] = typical_p
            
            outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode output
            generated_sequence = outputs.sequences[0]
            output_text = tokenizer.decode(generated_sequence[input_length:], skip_special_tokens=True)
            
            # Process scores (logits) to get probabilities and entropy
            # outputs.scores is a tuple of length max_new_tokens
            token_logprobs = []
            entropies = []
            
            for i, logits in enumerate(outputs.scores):
                # logits shape: (1, vocab_size)
                entropy = compute_entropy(logits)[0].item()
                entropies.append(entropy)
                
                # Get logprob of the actual generated token
                chosen_token_id = generated_sequence[input_length + i]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logprob = log_probs[0, chosen_token_id].item()
                token_logprobs.append(token_logprob)
                
            # Compute summary stats
            avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else 0.0
            avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            sequence_nll = -float(np.sum(token_logprobs)) if token_logprobs else 0.0
            
            result = {
                "prompt": prompt,
                "output": output_text,
                "token_logprobs": token_logprobs,
                "entropies": entropies,
                "entropy_stats": {
                    "avg_entropy": avg_entropy,
                    "max_entropy": float(np.max(entropies)) if entropies else 0.0,
                    "min_entropy": float(np.min(entropies)) if entropies else 0.0
                },
                "logprob_stats": {
                    "avg_logprob": avg_logprob,
                    "sequence_nll": sequence_nll
                },
                "params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "typical_p": typical_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed
                }
            }
            
            f.write(json.dumps(result) + "\n")
            print(f"Generated sample for prompt: {prompt[:30]}... | Avg LogProb: {avg_logprob:.3f} | Avg Entropy: {avg_entropy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--output", type=str, default="../results/baseline.jsonl")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--rep_pen", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Allow passing multiple prompts via a text file later, but for now just single prompt string
    prompts = [args.prompt]
    if os.path.exists(args.prompt):
        with open(args.prompt, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
    generate_samples(
        model_name=args.model,
        prompts=prompts,
        output_file=args.output,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        typical_p=args.typical_p,
        repetition_penalty=args.rep_pen,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
        device=args.device
    )
