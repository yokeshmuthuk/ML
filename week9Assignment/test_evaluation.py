"""
Test Set Evaluation Script
Evaluates trained models on test sets and calculates baseline comparisons
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def load_dataset(filename):
    """Load and encode a text dataset"""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    return data, vocab_size, encode, decode, chars

def calculate_baseline_loss(data, vocab_size):
    """
    Calculate loss for a baseline model that predicts uniform distribution
    This represents random guessing
    """
    # For uniform distribution over vocab_size categories
    # Cross-entropy loss = -log(1/vocab_size) = log(vocab_size)
    baseline_loss = math.log(vocab_size)
    return baseline_loss

def calculate_unigram_baseline(data, vocab_size):
    """
    Calculate loss for unigram baseline (predicts based on character frequency)
    """
    # Count character frequencies
    char_counts = torch.bincount(data, minlength=vocab_size).float()
    char_probs = char_counts / char_counts.sum()

    # Calculate cross-entropy loss for unigram model
    # For each character in data, look up its probability and take -log
    losses = []
    for char_idx in data:
        prob = char_probs[char_idx].item()
        if prob > 0:
            losses.append(-math.log(prob))
        else:
            losses.append(float('inf'))

    unigram_loss = sum(losses) / len(losses)
    return unigram_loss

@torch.no_grad()
def evaluate_model_on_dataset(model, data, block_size, batch_size=64, device='cpu', eval_iters=200):
    """
    Evaluate a trained model on a dataset
    """
    model.eval()
    losses = []

    for _ in range(eval_iters):
        # Sample random batch
        if len(data) <= block_size:
            print(f"Warning: Dataset too small ({len(data)}) for block_size ({block_size})")
            return float('nan')

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)

# Example usage (to be filled in with actual model loading)
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test datasets
    print("Loading datasets...")
    child_test_data, child_vocab_size, _, _, _ = load_dataset('input_childSpeech_testSet.txt')
    shakespeare_data, shakespeare_vocab_size, _, _, _ = load_dataset('input_shakespeare.txt')

    # Calculate baselines
    print("\n" + "="*60)
    print("BASELINE CALCULATIONS")
    print("="*60)

    print(f"\nChild Speech Test Set (vocab_size={child_vocab_size}):")
    child_uniform_baseline = calculate_baseline_loss(child_test_data, child_vocab_size)
    child_unigram_baseline = calculate_unigram_baseline(child_test_data, child_vocab_size)
    print(f"  Uniform baseline (random guessing): {child_uniform_baseline:.4f}")
    print(f"  Unigram baseline (frequency-based):  {child_unigram_baseline:.4f}")

    print(f"\nShakespeare Dataset (vocab_size={shakespeare_vocab_size}):")
    shakespeare_uniform_baseline = calculate_baseline_loss(shakespeare_data, shakespeare_vocab_size)
    shakespeare_unigram_baseline = calculate_unigram_baseline(shakespeare_data, shakespeare_vocab_size)
    print(f"  Uniform baseline (random guessing): {shakespeare_uniform_baseline:.4f}")
    print(f"  Unigram baseline (frequency-based):  {shakespeare_unigram_baseline:.4f}")

    print("\n" + "="*60)
    print("To evaluate your trained model:")
    print("1. Load your model checkpoint")
    print("2. Call evaluate_model_on_dataset() with your model")
    print("3. Compare against the baselines above")
    print("="*60)

    """
    # Example of how to use with a trained model:

    # Load model checkpoint
    checkpoint = torch.load('model_config3.pt')

    # Rebuild model architecture (must match training config)
    # ... create model instance ...
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)

    # Evaluate on test sets
    child_test_loss = evaluate_model_on_dataset(model, child_test_data, block_size=128, device=device)
    shakespeare_loss = evaluate_model_on_dataset(model, shakespeare_data, block_size=128, device=device)

    print(f"Model loss on child test set: {child_test_loss:.4f}")
    print(f"Model loss on Shakespeare: {shakespeare_loss:.4f}")
    """
