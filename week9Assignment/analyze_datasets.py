"""
Dataset Analysis Script
Analyzes the three datasets for vocabulary size, length, and content
"""

def analyze_dataset(filename):
    """Analyze a text dataset"""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get unique characters (vocabulary)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Get length
    total_length = len(text)

    # Get sample of content
    sample = text[:500] if len(text) > 500 else text

    # Count lines
    lines = text.split('\n')
    num_lines = len(lines)

    # Get character frequencies for top 10
    from collections import Counter
    char_freq = Counter(text)
    top_chars = char_freq.most_common(10)

    print(f"\n{'='*60}")
    print(f"Analysis of: {filename}")
    print(f"{'='*60}")
    print(f"Total length: {total_length:,} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Number of lines: {num_lines}")
    print(f"\nVocabulary: {repr(''.join(chars))}")
    print(f"\nTop 10 most frequent characters:")
    for char, count in top_chars:
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        print(f"  {char_display}: {count:,} ({count/total_length*100:.2f}%)")
    print(f"\nFirst 500 characters:")
    print(f"{'-'*60}")
    print(sample)
    print(f"{'-'*60}")

    return {
        'filename': filename,
        'total_length': total_length,
        'vocab_size': vocab_size,
        'num_lines': num_lines,
        'vocabulary': chars,
        'sample': sample
    }

if __name__ == "__main__":
    datasets = [
        'input_childSpeech_trainingSet.txt',
        'input_childSpeech_testSet.txt',
        'input_shakespeare.txt'
    ]

    results = {}
    for dataset in datasets:
        try:
            results[dataset] = analyze_dataset(dataset)
        except FileNotFoundError:
            print(f"\nError: Could not find {dataset}")

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Dataset':<40} {'Length':>10} {'Vocab':>8}")
    print(f"{'-'*60}")
    for dataset, info in results.items():
        print(f"{dataset:<40} {info['total_length']:>10,} {info['vocab_size']:>8}")
