# RNN Model Comparison

## Experiment: Vocabulary Size Impact on Text Generation Quality

### Model 1: 3K Vocabulary (BETTER MODEL)
**Configuration:**
- Vocabulary: 3,000 words
- LSTM Units: 256
- Dropout: 0.2
- Epochs trained: 12 (early stop)
- Patience: 8

**Results:**
- Train Accuracy: **15.85%**
- Val Accuracy: **11.99%**
- Train Perplexity: 76.31
- Val Perplexity: 345.70
- Overfitting Gap: 3.86%

**Generation Quality (Temperature 1.0):**
```
Seed: "wherefore art thou"
Output: "wherefore art thou there i'll not? my face are nothing to my tongue,
with when whither be thy wrong as hard it will these honour is thy that.
but late he bid me"
```

**Strengths:**
- ✓ Better validation accuracy (11.99%)
- ✓ Lower validation perplexity (345.70)
- ✓ More coherent sentence structure
- ✓ Better generalization (smaller overfitting gap)

---

### Model 2: 5K Vocabulary
**Configuration:**
- Vocabulary: 5,000 words
- LSTM Units: 256
- Dropout: 0.15 (less regularization)
- Epochs trained: 15 (early stop)
- Patience: 12

**Results:**
- Train Accuracy: **19.24%**
- Val Accuracy: **9.90%** ❌
- Train Perplexity: 56.55
- Val Perplexity: 1129.71 ❌
- Overfitting Gap: 9.34%

**Generation Quality (Temperature 1.0):**
```
Seed: "wherefore art thou"
Output: "wherefore art thou achilles which at us. like it gone that gave shall
and here comes yet. doth he at nay, myself when he, in. and canst now,
into the business on lord,"
```

**Weaknesses:**
- ✗ Worse validation accuracy (dropped from 11.99% → 9.90%)
- ✗ Much worse validation perplexity (345 → 1129)
- ✗ More fragmented sentences
- ✗ Severe overfitting (9.34% gap)

---

## Analysis

### Why Model 2 Performed Worse

**1. Vocabulary Too Large for Dataset Size**
- 5K vocabulary on ~900K words = only 180 occurrences per word on average
- Many rare words had insufficient training examples
- Model couldn't learn good representations for all words

**2. Lower Dropout Exacerbated Overfitting**
- Dropout 0.15 (vs 0.2) provided less regularization
- Combined with larger vocabulary → severe overfitting
- Train accuracy improved but val accuracy dropped

**3. Validation Perplexity Explosion**
- Val perplexity 1129 vs 345 indicates model is "confused"
- When encountering unseen word combinations, model makes very uncertain predictions
- Lower dropout meant model memorized training patterns rather than learning general language structure

### Key Insight: The Sweet Spot

**For ~900K word corpus:**
- ✓ 3K vocabulary is optimal
- ✓ Dropout 0.2 provides right balance
- ✗ 5K vocabulary overstretches model capacity

### Recommendations

**For better performance:**
1. **Use Model 1 (3K vocab)** for deployment
2. To improve Model 2, would need:
   - Increase training data (2-3x more text)
   - Increase model size (384+ LSTM units)
   - Increase dropout back to 0.2-0.25
   - Train much longer (30+ epochs)

---

## Saved Models

- `model_3kvocab_256units_12epochs.pt` - **RECOMMENDED**
- `model_5kvocab_256units_15epochs.pt` - Overfitted, not recommended

---

## Assignment Deliverable

This comparison demonstrates:
1. ✓ **Hyperparameter experimentation** (vocabulary size)
2. ✓ **Quantitative analysis** (accuracy, perplexity metrics)
3. ✓ **Qualitative analysis** (text generation samples)
4. ✓ **Understanding of overfitting** and regularization
5. ✓ **Data-driven conclusions** about optimal configuration

Use this for your **Section 4: Experiments & Results** in the technical report.
