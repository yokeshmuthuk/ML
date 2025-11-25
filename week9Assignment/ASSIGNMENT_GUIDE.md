# Week 9 Assignment Guide
## Transformer Model Experiments with nanoGPT

This guide walks you through completing the assignment with all the prepared scripts.

---

## Part (i): Dataset Analysis & Model Configuration

### (i.a) Dataset Description

Run: `python analyze_datasets.py`

**Results Summary:**

1. **input_childSpeech_trainingSet.txt**
   - Length: 246,982 characters
   - Vocabulary: 40 unique characters (simple child speech vocabulary)
   - Content: Simple sentences like "I want cookie", "Night night", "More more"
   - Character set: `\n ABCDGHILMNRSUWYabcdefghijklmnoprstuvwy`
   - Domain: Child language acquisition, simple grammatical structures

2. **input_childSpeech_testSet.txt**
   - Length: 24,113 characters (10% of training set)
   - Vocabulary: 40 unique characters (same as training)
   - Content: Similar to training set (held-out data)
   - Purpose: Evaluate generalization on same domain

3. **input_shakespeare.txt**
   - Length: 1,115,394 characters (4.5x larger)
   - Vocabulary: 65 unique characters (includes punctuation, capitals)
   - Content: Shakespearean plays with complex vocabulary
   - Domain: Formal English, complex sentence structures
   - Purpose: Out-of-distribution evaluation

**Key Observations:**
- Child speech has smaller vocabulary (40 vs 65)
- Shakespeare is much larger and more complex
- Different linguistic complexity levels
- Test ability to generalize to different domains

---

### (i.b-c) Model Configurations (<1M Parameters)

Run: `python calculate_parameters.py` to verify parameter counts

#### Configuration 1: Aggressive Reduction (0.82M params)
**File:** `gpt_config1.py`

**Hyperparameters:**
```python
n_embd = 128      # Reduced from 384
n_head = 4        # Reduced from 6
n_layer = 4       # Reduced from 6
block_size = 128  # Reduced from 256
```

**Rationale:**
- **Primary reduction: n_embd (384→128)** - Embedding dimension has quadratic impact on feed-forward network parameters (FFN uses 4*n_embd hidden units)
- Reduces both token embeddings AND FFN parameters significantly
- Maintains reasonable head_size (128/4 = 32) for attention
- Smaller block_size reduces positional embeddings and context window
- Most aggressive approach: smallest model, fastest training

**Expected Characteristics:**
- Faster training, less memory
- May underfit due to limited capacity
- Simple patterns should be learnable

---

#### Configuration 2: Fewer Layers, Higher Embedding (0.95M params)
**File:** `gpt_config2.py`

**Hyperparameters:**
```python
n_embd = 192      # Moderate reduction from 384
n_head = 4        # Reduced from 6
n_layer = 3       # Halved from 6
block_size = 128  # Reduced from 256
```

**Rationale:**
- **Trade depth for width:** Fewer transformer blocks but richer representations
- Hypothesis: Child speech is simple and may not need deep hierarchical processing
- Higher embedding dimension (192) allows more expressive feature learning
- Each layer can learn more complex patterns with larger embeddings
- Suitable for tasks where pattern complexity > hierarchical structure

**Expected Characteristics:**
- Better representation capacity than Config 1
- May struggle with long-range dependencies (only 3 layers)
- Good middle ground for simple language

---

#### Configuration 3: Balanced Approach (0.75M params)
**File:** `gpt_config3.py`

**Hyperparameters:**
```python
n_embd = 160      # Balanced reduction
n_head = 4        # Reduced from 6
n_layer = 3       # Halved from 6
block_size = 128  # Reduced from 256
```

**Rationale:**
- **Balanced reduction across all dimensions**
- Middle ground between Config 1 (very small) and Config 2 (wider)
- 160 embedding allows decent representation while keeping parameters low
- 3 layers provide some depth without excessive parameters
- Most "balanced" design: neither too shallow nor too narrow

**Expected Characteristics:**
- Good balance of capacity and efficiency
- Should perform reasonably well on child speech
- Likely best generalization due to balanced architecture

---

### Training the Configurations

```bash
# Train all three configurations
python gpt_config1.py  # Creates model_config1.pt
python gpt_config2.py  # Creates model_config2.pt
python gpt_config3.py  # Creates model_config3.pt
```

Each script will output:
- Parameter count verification
- Training/validation loss every 100 steps
- Generated text sample at the end

**Expected Training Results:**

*Training Loss Trends:*
- All configs should show decreasing loss
- Config 1 may plateau higher (limited capacity)
- Config 2/3 should achieve lower training loss

*Validation Loss:*
- Watch for overfitting (val loss > train loss)
- Config 1 less likely to overfit (smaller capacity)
- Config 2/3 may overfit if trained too long

*Qualitative Output:*
- Early iterations: gibberish
- Mid-training: word-like patterns, some real words
- Late training: grammatical simple sentences similar to dataset

---

### (i.d) Bias Terms in Self-Attention

**File:** `gpt_bias_experiment.py`

**Experiment:** Compare `bias=False` vs `bias=True` in attention layers (Q, K, V projections)

**Run:**
```bash
# First: Set USE_BIAS_IN_ATTENTION = False
python gpt_bias_experiment.py  # Creates model_bias_False.pt

# Then: Set USE_BIAS_IN_ATTENTION = True
python gpt_bias_experiment.py  # Creates model_bias_True.pt
```

**What to Report:**
1. Parameter count difference (bias adds 3*head_size parameters per head)
2. Training/validation loss comparison
3. Generated text quality
4. Convergence speed

**Expected Findings:**
- Bias=True: Slightly more parameters, may converge faster
- Bias=False: Slightly regularized, forces model to learn relative patterns
- For simple data like child speech, difference may be minimal
- Bias terms can help with centering/shifting attention scores

**Discussion Points:**
- Bias in attention allows learned offset to attention scores
- Without bias, attention purely based on relative similarity (Q·K^T)
- Modern transformers (GPT-2/3) typically use bias=True for flexibility
- Trade-off: slight parameter increase vs. modeling flexibility

---

### (i.e) Skip Connections Impact

**File:** `gpt_skip_experiment.py`

**Experiment:** Compare transformer WITH vs WITHOUT residual connections

**Run:**
```bash
# First: Set USE_SKIP_CONNECTIONS = True
python gpt_skip_experiment.py  # Creates model_skip_True.pt

# Then: Set USE_SKIP_CONNECTIONS = False
python gpt_skip_experiment.py  # Creates model_skip_False.pt
```

**What to Report:**
1. Training loss curves (WITH skip should train much better)
2. Validation loss comparison
3. Gradient flow analysis (optional: check gradient norms)
4. Generated text quality

**Expected Findings:**
- **WITH skip connections:** Stable training, good performance
- **WITHOUT skip connections:**
  - May suffer from vanishing gradients
  - Training may be unstable or very slow
  - Higher final loss
  - Poor text generation quality

**Discussion Points:**
- Skip connections enable gradient flow through deep networks
- Without them: gradients vanish in deeper layers (3+ layers)
- Residual connections: `x = x + F(x)` allows identity mapping
- Critical for training deep transformers (essential architectural component)
- Enables learning of residual/refinement functions rather than full transformation

---

## Part (ii): Test Set Evaluation

### Baseline Calculations

Run: `python test_evaluation.py`

This calculates:
1. **Uniform baseline:** Random guessing (log(vocab_size))
2. **Unigram baseline:** Frequency-based prediction

**Expected Baselines:**
- Child speech (40 chars): ~3.69 (uniform), ~2.5-3.0 (unigram)
- Shakespeare (65 chars): ~4.17 (uniform), ~2.8-3.5 (unigram)

---

### (ii.a) Test on Child Speech Test Set

**What to do:**
1. Select best model from Part (i) based on validation loss
2. Load model and evaluate on `input_childSpeech_testSet.txt`
3. Compare to baselines

**Expected Results:**
- Test loss should be close to validation loss (if not overfitting)
- Should be MUCH better than uniform baseline (< 3.69)
- Should beat unigram baseline (< 2.5-3.0)
- Typical good result: 1.5-2.5 range

**Interpretation:**
- **Good:** Loss significantly below baselines → model learned language patterns
- **Bad:** Loss near or above baselines → model didn't learn
- Test vs. Validation comparison:
  - Similar: Good generalization
  - Test >> Val: Overfitting
  - Test < Val: Lucky (or val set has harder examples)

---

### (ii.b) Test on Shakespeare Dataset

**What to do:**
1. Use same best model (trained on child speech)
2. Evaluate on Shakespeare text
3. Compare to child speech test loss

**Expected Results:**
- Loss will be HIGHER than child speech test
- May be close to Shakespeare baselines (poor performance expected)
- Demonstrates domain shift problem

**Why Higher Loss?**
1. **Vocabulary mismatch:** Shakespeare has 65 chars, model trained on 40
   - Characters like punctuation may get mapped incorrectly
2. **Domain shift:** Different linguistic patterns
   - Complex vs. simple grammar
   - Formal vs. child language
3. **Distributional difference:** Character frequencies differ

**Interpretation & Practical Applications:**

*What this tells us:*
- Models are domain-specific
- Transfer learning requires fine-tuning or adaptation
- Out-of-distribution detection is possible via loss

*Practical uses:*
1. **Anomaly Detection:** High loss indicates out-of-domain input
2. **Data Quality Monitoring:** Sudden loss spikes suggest distribution shift
3. **Domain Adaptation:** Identifies when fine-tuning is needed
4. **Perplexity-based Filtering:** Filter training data by loss threshold
5. **Active Learning:** Select high-loss examples for labeling

---

## Running Instructions

### Prerequisites
```bash
pip3 install torch --break-system-packages
```

### Complete Workflow

```bash
# 1. Analyze datasets
python analyze_datasets.py

# 2. Calculate parameters for configs
python calculate_parameters.py

# 3. Train all configurations
python gpt_config1.py
python gpt_config2.py
python gpt_config3.py

# 4. Bias experiment
# Edit gpt_bias_experiment.py: set USE_BIAS_IN_ATTENTION = False, then run
python gpt_bias_experiment.py
# Edit again: set USE_BIAS_IN_ATTENTION = True, then run
python gpt_bias_experiment.py

# 5. Skip connection experiment
# Edit gpt_skip_experiment.py: set USE_SKIP_CONNECTIONS = False, then run
python gpt_skip_experiment.py
# Edit again: set USE_SKIP_CONNECTIONS = True, then run
python gpt_skip_experiment.py

# 6. Calculate baselines
python test_evaluation.py

# 7. Evaluate models on test sets
# (You'll need to modify test_evaluation.py to load your best model)
```

---

## Report Writing Tips

### For Each Question:

**Don't just say:** "The loss was 2.34"

**Do say:** "The validation loss converged to 2.34, which is significantly better than the unigram baseline of 3.02, indicating the model learned meaningful patterns. However, it's higher than Config 2's loss of 2.10, suggesting the larger embedding dimension in Config 2 provided better representation capacity for this task."

### Key Points to Cover:

1. **Dataset Analysis:** Describe content, vocab, and what makes each dataset unique
2. **Configuration Rationale:** Explain WHY you chose each reduction
3. **Loss Comparison:** Compare across configs AND to baselines
4. **Overfitting Analysis:** Compare train vs. validation loss
5. **Qualitative Assessment:** Describe generated text quality
6. **Architectural Impact:** Explain technical reasons for differences
7. **Interpretation:** What do the numbers mean? Why do we care?

### Plot Requirements:
- Label axes clearly
- Large, legible fonts
- Include legend
- Caption explaining what it shows

---

## Expected Completion Checklist

- [ ] Dataset analysis for all 3 datasets
- [ ] 3 model configurations trained with rationale explained
- [ ] Loss curves plotted and compared
- [ ] Bias experiment completed (2 runs)
- [ ] Skip connection experiment completed (2 runs)
- [ ] Test evaluation on child speech with baseline comparison
- [ ] Test evaluation on Shakespeare with interpretation
- [ ] Report written (5-10 pages)
- [ ] Code appended to report
- [ ] ZIP file created with all code and data

---

Good luck with your assignment!
