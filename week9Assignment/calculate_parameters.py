"""
Calculate the number of parameters in the GPT model
Helps determine how to downsize to < 1M parameters
"""

def calculate_params(vocab_size, block_size, n_embd, n_head, n_layer):
    """
    Calculate total parameters in GPT model

    Args:
        vocab_size: Size of vocabulary
        block_size: Maximum context length
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_layer: Number of transformer layers
    """

    # Token and position embeddings
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd

    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"{'='*60}")
    print(f"vocab_size: {vocab_size}")
    print(f"block_size: {block_size}")
    print(f"n_embd: {n_embd}")
    print(f"n_head: {n_head}")
    print(f"n_layer: {n_layer}")

    print(f"\n{'='*60}")
    print(f"Parameter Breakdown:")
    print(f"{'='*60}")
    print(f"Token embeddings: {token_emb:,} ({vocab_size} * {n_embd})")
    print(f"Position embeddings: {pos_emb:,} ({block_size} * {n_embd})")

    # Per transformer block
    head_size = n_embd // n_head

    # Multi-head attention:
    # Q, K, V projections: 3 * (n_embd * head_size) per head
    # Output projection: (head_size * n_head) * n_embd
    qkv_params = 3 * n_head * (n_embd * head_size)
    proj_params = (head_size * n_head) * n_embd
    attn_params = qkv_params + proj_params

    # Feed-forward network:
    # First layer: n_embd * (4 * n_embd)
    # Second layer: (4 * n_embd) * n_embd
    ff_params = n_embd * (4 * n_embd) + (4 * n_embd) * n_embd

    # Layer norms (2 per block, each has 2*n_embd parameters for weight and bias)
    ln_params = 2 * 2 * n_embd

    # Total per block
    block_params = attn_params + ff_params + ln_params

    print(f"\nPer Transformer Block:")
    print(f"  Multi-head attention: {attn_params:,}")
    print(f"    - Q, K, V projections: {qkv_params:,}")
    print(f"    - Output projection: {proj_params:,}")
    print(f"  Feed-forward: {ff_params:,}")
    print(f"    - First layer: {n_embd * (4 * n_embd):,}")
    print(f"    - Second layer: {(4 * n_embd) * n_embd:,}")
    print(f"  Layer norms: {ln_params:,}")
    print(f"  Total per block: {block_params:,}")

    # All blocks
    all_blocks = n_layer * block_params
    print(f"\nAll {n_layer} blocks: {all_blocks:,}")

    # Final layer norm and output head
    final_ln = 2 * n_embd
    lm_head = n_embd * vocab_size

    print(f"\nFinal layers:")
    print(f"  Final layer norm: {final_ln:,}")
    print(f"  LM head (output): {lm_head:,}")

    # Total
    total = token_emb + pos_emb + all_blocks + final_ln + lm_head

    print(f"\n{'='*60}")
    print(f"TOTAL PARAMETERS: {total:,} ({total/1e6:.4f}M)")
    print(f"{'='*60}\n")

    return total

if __name__ == "__main__":
    # Original configuration (with Shakespeare dataset)
    print("\n" + "="*60)
    print("ORIGINAL CONFIGURATION (Shakespeare)")
    print("="*60)
    original = calculate_params(vocab_size=65, block_size=256, n_embd=384, n_head=6, n_layer=6)

    # Child speech dataset has smaller vocab
    print("\n" + "="*60)
    print("ORIGINAL CONFIG WITH CHILD SPEECH DATASET")
    print("="*60)
    child_original = calculate_params(vocab_size=40, block_size=256, n_embd=384, n_head=6, n_layer=6)

    print("\n" + "="*60)
    print("EXPLORING DOWNSIZING OPTIONS")
    print("="*60)

    # Configuration 1: Reduce embedding dimension
    print("\n--- CONFIG 1: Reduce n_embd (384 -> 192) ---")
    config1 = calculate_params(vocab_size=40, block_size=256, n_embd=192, n_head=6, n_layer=6)

    # Configuration 2: Reduce number of layers
    print("\n--- CONFIG 2: Reduce n_layer (6 -> 3) ---")
    config2 = calculate_params(vocab_size=40, block_size=256, n_embd=384, n_head=6, n_layer=3)

    # Configuration 3: Reduce both embedding and heads
    print("\n--- CONFIG 3: Reduce n_embd (384 -> 256) and n_head (6 -> 4) ---")
    config3 = calculate_params(vocab_size=40, block_size=256, n_embd=256, n_head=4, n_layer=6)

    # Configuration 4: Reduce embedding and layers
    print("\n--- CONFIG 4: Reduce n_embd (384 -> 256) and n_layer (6 -> 4) ---")
    config4 = calculate_params(vocab_size=40, block_size=256, n_embd=256, n_head=4, n_layer=4)

    # Configuration 5: Aggressive reduction
    print("\n--- CONFIG 5: Reduce n_embd (384 -> 128), n_head (6 -> 4), n_layer (6 -> 4) ---")
    config5 = calculate_params(vocab_size=40, block_size=128, n_embd=128, n_head=4, n_layer=4)
