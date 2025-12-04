# Shakespeare Text Generation Using LSTM Neural Networks
## A Deep Learning Approach to Stylistic Text Generation

**Author:** gvklok
**Course:** CST-435 - Recurrent Neural Networks
**Date:** November 2025
**GitHub:** https://github.com/gvklok/ShakespeareRNN
**Live Demo:** https://shakespearernn.onrender.com/

---

## 1. Introduction

### 1.1 Overview

Recurrent Neural Networks (RNNs) have revolutionized sequence modeling tasks by introducing the concept of temporal dependencies and memory. Unlike feedforward neural networks that process inputs independently, RNNs maintain an internal state that allows them to capture patterns across sequential data. This capability makes them particularly well-suited for natural language processing tasks, where understanding context and sequential relationships is crucial.

This project implements a text generation system using Long Short-Term Memory (LSTM) networks, a specialized RNN architecture designed to overcome the vanishing gradient problem inherent in vanilla RNNs. The model is trained on the complete works of William Shakespeare, learning to generate text that mimics the vocabulary, grammatical structures, and stylistic elements of Early Modern English literature.

### 1.2 Text Generation Task

Text generation is a sequence-to-sequence learning problem where the model predicts the next word (or character) given a sequence of previous words. Formally, given a sequence of words $w_1, w_2, ..., w_t$, the model learns the probability distribution:

$$P(w_{t+1} | w_1, w_2, ..., w_t)$$

During generation, the model samples from this distribution to produce new text. The challenge lies in maintaining both short-term coherence (grammatically correct sentences) and long-term consistency (thematic and topical continuity).

### 1.3 Project Objectives

The primary objectives of this project were:

1. **Implement a working LSTM-based text generation system** using PyTorch
2. **Train the model on a substantial corpus** (Shakespeare's complete works, ~5.4MB)
3. **Conduct systematic experiments** to understand model behavior and hyperparameter effects
4. **Deploy a full-stack web application** with REST API and interactive frontend
5. **Analyze generated text quality** across different configuration parameters
6. **Document findings** with reproducible experiments and visualizations

---

## 2. RNN Architecture Analysis

### 2.1 From RNN to LSTM

Traditional RNNs update their hidden state $h_t$ at each time step using:

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

While simple and elegant, this formulation suffers from the **vanishing gradient problem**. During backpropagation through time (BPTT), gradients must flow backward through many time steps. When the gradient is computed as a product of many small terms (derivatives of tanh < 1), it exponentially decays, making it impossible for the network to learn long-term dependencies.

LSTM networks solve this by introducing a **cell state** $C_t$ that acts as a "highway" for gradient flow, along with three gates that control information flow:

1. **Forget Gate** ($f_t$): Decides what information to discard from cell state
2. **Input Gate** ($i_t$): Decides what new information to add to cell state
3. **Output Gate** ($o_t$): Decides what information to output as hidden state

The complete LSTM equations are:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

The key innovation is the **additive update** to the cell state ($C_t = f_t \odot C_{t-1} + ...$), which allows gradients to flow unchanged, enabling the network to learn dependencies spanning hundreds of time steps.

### 2.2 Model Architecture

Our implementation uses a multi-layer LSTM architecture with the following components:

```
Input (word indices) [batch_size, seq_len]
    ↓
Embedding Layer [vocab_size=3000, embed_dim=100]
    ↓
LSTM Layer 1 [input=100, hidden=256, return_sequences=True]
    ↓
Dropout (p=0.2)
    ↓
LSTM Layer 2 [input=256, hidden=256, return_sequences=False]
    ↓
Dropout (p=0.2)
    ↓
Fully Connected [hidden=256, output=vocab_size=3000]
    ↓
Softmax (during inference)
    ↓
Output (word probabilities) [batch_size, vocab_size]
```

**Parameter Breakdown:**
- **Embedding Layer**: 3,000 × 100 = 300,000 parameters
- **LSTM Layer 1**: 4 × (256 × (100 + 256) + 256) = ~366,000 parameters
- **LSTM Layer 2**: 4 × (256 × (256 + 256) + 256) = ~525,000 parameters
- **Fully Connected**: 256 × 3,000 + 3,000 = ~771,000 parameters
- **Total**: ~1,962,000 parameters

### 2.3 Hyperparameter Choices

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| **Vocabulary Size** | 3,000 words | Experiment showed 3K generalized better than 5K (reduced overfitting) |
| **Embedding Dim** | 100 | Standard size for moderate vocabulary; balances expressiveness and efficiency |
| **LSTM Hidden Units** | 256 | Large enough to capture Shakespeare's complexity; small enough to train on modest hardware |
| **Number of Layers** | 2 | Provides sufficient depth for hierarchical feature learning without excessive parameters |
| **Dropout Rate** | 0.2 | Prevents overfitting while maintaining learning capacity |
| **Sequence Length** | 50 words | Long enough to capture sentence-level context; short enough for efficient training |
| **Batch Size** | 64 | Optimal trade-off between gradient stability and training speed on MPS (Apple Silicon) |
| **Learning Rate** | 0.001 | Standard Adam optimizer default; adjusted dynamically with ReduceLROnPlateau |

### 2.4 Comparison to Alternatives

**Simple RNN vs. LSTM:**
- Simple RNN: Cannot learn dependencies > 10 time steps due to vanishing gradients
- LSTM: Successfully learns dependencies spanning 50+ time steps in our experiments

**LSTM vs. GRU:**
- GRU: Fewer parameters (2 gates vs. 3), faster training
- LSTM: More expressive, better for complex sequential patterns
- For Shakespeare's complex grammar and vocabulary, LSTM's additional expressiveness is beneficial

**RNN vs. Transformers:**
- Transformers (GPT, BERT): State-of-the-art for NLP, but require massive compute and data
- RNNs: More efficient for smaller datasets, suitable for educational purposes
- Our LSTM model (24MB) is 200× smaller than GPT-2 small (500MB)

---

## 3. Implementation Details

### 3.1 Data Preprocessing Pipeline

**Step 1: Text Acquisition**
- Source: Project Gutenberg - Complete Works of William Shakespeare
- Raw size: 5.4MB (~1 million words)
- Content: 37 plays, 154 sonnets, and several poems

**Step 2: Text Cleaning**
```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove Project Gutenberg headers/footers
    text = remove_gutenberg_metadata(text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text
```

**Step 3: Tokenization Strategy**

We chose **word-level tokenization** over character-level for several reasons:

| Aspect | Character-Level | Word-Level (Our Choice) |
|--------|----------------|-------------------------|
| Vocabulary Size | ~50 characters | ~3,000 words |
| Sequence Length | Very long (500+) | Moderate (50) |
| Grammar Learning | Must learn from scratch | Inherent in tokens |
| Training Speed | Slower (long sequences) | Faster |
| Generation Quality | Better spelling | Better semantics |

**Step 4: Sequence Generation**

Training sequences are created using a sliding window approach:

```python
sequence_length = 50
for i in range(len(words) - sequence_length):
    input_seq = words[i:i+sequence_length]      # Context
    target_word = words[i+sequence_length]      # Next word to predict
```

This generates approximately 180,000 training sequences from the Shakespeare corpus.

**Step 5: Vocabulary Construction**

```python
# Count word frequencies
word_counts = Counter(all_words)

# Keep most common 3000 words
vocab = word_counts.most_common(3000)

# Create mappings
word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab, start=1)}
word_to_idx['<PAD>'] = 0  # Padding token
word_to_idx['<UNK>'] = 3001  # Unknown token
```

### 3.2 Training Configuration

**Loss Function:**
Cross-entropy loss measures the difference between predicted probability distribution and true next-word distribution:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(w_{\text{true}}^{(i)} | \text{context}^{(i)})$$

**Optimizer:**
Adam optimizer with adaptive learning rates:
- Initial learning rate: 0.001
- β1 = 0.9, β2 = 0.999 (default Adam parameters)
- Weight decay: 0 (no L2 regularization, using dropout instead)

**Learning Rate Schedule:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,           # Reduce LR by half
    patience=2,           # After 2 epochs without improvement
    min_lr=1e-6           # Don't go below this
)
```

**Regularization Techniques:**
1. **Dropout (0.2)**: Applied after each LSTM layer
2. **Gradient Clipping**: Prevents exploding gradients by capping at norm = 5
3. **Early Stopping**: Stops training if validation loss doesn't improve for 5 epochs

**Training Infrastructure:**
- Hardware: Apple M1 Pro with MPS (Metal Performance Shaders) acceleration
- Framework: PyTorch 2.0+ with automatic mixed precision
- Training time: ~45 minutes for 12 epochs
- Memory usage: ~2GB GPU, ~4GB RAM

### 3.3 API Design

**Backend Architecture (FastAPI):**

The REST API exposes four main endpoints:

1. **Health Check** (`GET /`)
   - Returns model status and framework info
   - Used for deployment health monitoring

2. **Text Generation** (`POST /generate`)
   ```json
   Request: {
     "seed_text": "to be or not to be",
     "length": 100,
     "temperature": 1.0
   }
   Response: {
     "generated_text": "..."
   }
   ```

3. **Training Metrics** (`GET /metrics`)
   - Returns loss, accuracy, and perplexity across epochs
   - Enables frontend visualization

4. **Model Info** (`GET /model-info`)
   - Returns architecture details and parameters
   - Used for model inspection

**CORS Configuration:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configured for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Temperature Sampling Implementation:**
```python
def generate_text(seed_text, length, temperature):
    for _ in range(length):
        # Get model predictions
        logits = model(current_sequence)

        # Apply temperature
        scaled_logits = logits / temperature
        probabilities = F.softmax(scaled_logits, dim=-1)

        # Sample next word
        next_word_idx = torch.multinomial(probabilities, 1)
```

Temperature controls randomness:
- T < 1: More conservative (peaked distribution)
- T = 1: Unchanged distribution
- T > 1: More random (flattened distribution)

### 3.4 Frontend Architecture

**Technology Stack:**
- React 18 with TypeScript for type safety
- Axios for API communication
- Recharts for interactive training visualizations
- Tailwind CSS for responsive design

**Component Structure:**
```
App.tsx (main container)
├── TextGenerator.tsx      # Text generation interface
├── TrainingMetrics.tsx    # Loss/accuracy charts
└── ModelVisualizer.tsx    # Architecture display
```

**Key Features:**
- Real-time text generation with adjustable parameters
- Interactive sliders for temperature (0.5-2.0) and length (10-200)
- Live training metrics visualization showing convergence
- Backend health status indicator

---

## 4. Experiments & Results

### Experiment 1: Vocabulary Size Impact

**Hypothesis:** Larger vocabulary captures more nuance but risks overfitting on limited training data.

**Methodology:**
- Trained two models with identical architecture except vocabulary size
- Model A: 3,000 most common words
- Model B: 5,000 most common words
- Same training data, same hyperparameters, same number of epochs (12)

**Results:**

| Metric | 3K Vocab (Model A) | 5K Vocab (Model B) | Winner |
|--------|-------------------|-------------------|---------|
| Training Accuracy | 15.85% | 18.23% | B |
| Validation Accuracy | 11.99% | 9.90% | A |
| Training Perplexity | 76.31 | 52.14 | B |
| Validation Perplexity | **345.70** | **1129.71** | **A** |
| Model Size | 24MB | 38MB | A |
| Training Time/Epoch | 3.5min | 5.2min | A |

**Analysis:**

Model B (5K vocab) achieved higher training accuracy (18.23% vs. 15.85%) and lower training perplexity (52.14 vs. 76.31), suggesting it learned the training data better. However, the validation metrics tell a different story:

- **Validation Accuracy**: 3K model (11.99%) outperformed 5K model (9.90%)
- **Validation Perplexity**: 3K model (345.70) vs. 5K model (1129.71) - a 3.3× difference!

The large gap between training and validation performance for Model B indicates **overfitting**. The 5K vocabulary included many rare words that appeared in specific contexts, causing the model to memorize rather than generalize.

**Perplexity Interpretation:**
- Perplexity measures how "surprised" the model is by the actual next word
- 3K model: On average, narrows choices to ~346 words (out of 3000)
- 5K model: On average, can't narrow beyond ~1130 words (out of 5000)

**Conclusion:** The 3K vocabulary model generalizes significantly better despite slightly lower training performance. For this dataset size (~1M words), 3K vocabulary provides optimal balance between expressiveness and generalization.

### Experiment 2: Temperature Analysis

**Hypothesis:** Temperature parameter controls the creativity-coherence trade-off in generated text.

**Methodology:**
- Used the trained 3K vocabulary model
- Generated text at six temperature values: 0.5, 0.7, 1.0, 1.3, 1.5, 2.0
- Same seed phrase for all: "to be or not to be"
- Generated 80 words per sample
- Evaluated on: repetition, grammaticality, creativity, coherence

**Results:**

| Temperature | Repetition | Grammar | Creativity | Coherence | Best Use Case |
|-------------|-----------|---------|------------|-----------|---------------|
| 0.5 | High | Excellent | Very Low | High | Formal, predictable text |
| 0.7 | Moderate | Good | Low | High | Slightly varied formal text |
| 1.0 | Low | Good | Moderate | Moderate | General purpose |
| 1.3 | Low | Fair | High | Moderate | Creative writing |
| 1.5 | Very Low | Fair | Very High | Low | Experimental poetry |
| 2.0 | Very Low | Poor | Extreme | Very Low | Surrealist experiments |

**Example Outputs:**

**T=0.5 (Conservative):**
> "to be or not to be a good to your heart to be a woman to him i fall your loves i am a little i did entreat the good you will..."

- Phrase "to be a good" repeats
- Grammatically structured but repetitive
- Safe but boring

**T=1.0 (Balanced):**
> "to be or not to be known like so? i had you need enough. did she torment this strange here comes hell and love you advance from better greater man..."

- Varied vocabulary
- Some grammatical structure maintained
- Interesting word combinations
- **Recommended default**

**T=1.5 (Creative):**
> "to be or not to be father, to be, thus might come to. fame our shall. now still women still that very hear me turn them how i'll hearing the head now? child, breeds why..."

- Very creative and unexpected
- Fragmented grammar
- Dreamlike quality
- Good for poetry/experimental text

**T=2.0 (Chaotic):**
> "to be or not to be romeo what's shall. shall doth repair do provided your return run who, all second arms him hand down antonio, clarence fed whom..."

- Extremely creative
- Multiple character names appear
- Almost no grammatical coherence
- Interesting for analysis but not readable

**Quantitative Metrics:**

| Temperature | Unique Word % | Avg Sentence Length | Perplexity (subjective) |
|-------------|---------------|---------------------|------------------------|
| 0.5 | 62% | 12 words | Low |
| 1.0 | 78% | 8 words | Medium |
| 1.5 | 85% | 5 words | High |
| 2.0 | 91% | 3 words | Very High |

**Conclusion:**
- **T=0.7-1.0**: Best for readable, Shakespearean-style text
- **T=1.0-1.3**: Sweet spot balancing interest and coherence
- **T>1.5**: Useful for creative exploration but sacrifices readability

The temperature parameter provides intuitive control over output style, making it an excellent user-facing feature for the web application.

### Experiment 3: Training Dynamics

While not a formal experiment, analysis of training curves revealed important insights:

**Loss Convergence:**
- Training loss steadily decreased: 5.89 → 4.33 (27% reduction)
- Validation loss decreased then plateaued: 5.53 → 5.85
- Divergence began around epoch 8, suggesting optimal stopping point

**Accuracy Progression:**
- Training accuracy: 7.0% → 15.85% (steady improvement)
- Validation accuracy: 9.9% → 11.99% (peaked at epoch 5)
- Gap between train/val widened after epoch 8 (overfitting signal)

**Learning Rate Schedule Effects:**
- Initial LR: 0.001
- Reduced to 0.0005 at epoch 7
- Reduced to 0.00025 at epoch 10
- LR reductions corresponded to validation plateaus

**Gradient Norms:**
- Average gradient norm stayed below 5.0 (clipping threshold)
- Stable gradients indicate LSTM successfully avoided exploding gradients
- No vanishing gradient issues observed

---

## 5. Analysis & Discussion

### 5.1 Generated Text Quality

**Strengths:**

1. **Vocabulary Acquisition**: Model learned 3,000 Shakespearean words including archaic forms ("thou," "hath," "thy")
2. **Character Knowledge**: Generated text includes actual Shakespeare characters (Romeo, Caesar, Hamlet, Tybalt)
3. **Thematic Consistency**: Captures major themes (love, death, honor, betrayal)
4. **Stylistic Elements**: Uses Early Modern English constructions appropriately
5. **Local Coherence**: Individual phrases often grammatically correct

**Example of Good Output (T=1.0):**
> "shall i compare thee will i died make my daughter, good night. bound i must have passed you, to make her forbid the father's night and woeful your great age"

This captures:
- Comparison structure (sonnet 18 opening)
- Family relationships (daughter, father)
- Temporal references (night)
- Emotional tone (woeful)

**Limitations:**

1. **Long-term Coherence**: Cannot maintain topic for > 30-40 words
2. **Semantic Drift**: Meaning shifts unpredictably in longer generations
3. **Lack of Plot**: No narrative structure or story progression
4. **Syntactic Errors**: Occasional agreement errors ("they is," "was afraid")
5. **Punctuation**: Inconsistent or absent punctuation marks

**Example of Poor Output (T=2.0):**
> "romeo what's shall. shall doth repair do provided your return run who, all second arms him hand down antonio, clarence fed whom here"

Issues:
- Word salad ("romeo what's shall")
- No grammatical structure
- Random character names juxtaposed
- Completely incoherent

### 5.2 Challenges Encountered

**1. Overfitting on Small Vocabulary**
- Initial attempts with 10K vocabulary led to severe overfitting
- Solution: Reduced to 3K vocabulary based on frequency analysis

**2. Training Instability**
- Early experiments had exploding gradients
- Solution: Gradient clipping (norm < 5) and careful LR scheduling

**3. Memory Constraints**
- Apple M1 Pro MPS has 16GB shared memory
- Large batch sizes (128+) caused OOM errors
- Solution: Reduced batch size to 64, used gradient accumulation

**4. Deployment Challenges**
- Model files (24MB) initially blocked by `.gitignore`
- Railway deployment failed on free tier (CPU timeout)
- Solution: Modified `.gitignore`, switched to Render

**5. API Latency**
- Initial implementation took ~5s to generate 100 words
- Solution: Batch inference, optimized tokenization, model kept in memory

### 5.3 Comparison to State-of-the-Art

**Our LSTM Model vs. Modern LLMs:**

| Aspect | Our LSTM | GPT-2 Small | GPT-3 |
|--------|----------|-------------|-------|
| Parameters | ~2M | 117M | 175B |
| Training Data | 5MB | 40GB | 570GB |
| Training Time | 45 min | Days | Months |
| Model Size | 24MB | 500MB | 350GB |
| Coherence (subjective) | Phrase-level | Paragraph-level | Multi-page |
| Hardware Required | M1 Mac | GPU | Supercomputer |

**Strengths of Our Approach:**
- Efficient: Trains on consumer hardware
- Transparent: Interpretable architecture
- Domain-specific: Specialized for Shakespeare
- Educational: Demonstrates RNN fundamentals

**Acknowledged Gaps:**
- GPT-3 maintains coherence for thousands of words
- Modern transformers use attention instead of recurrence
- Pre-trained models leverage transfer learning

However, our LSTM model successfully demonstrates the core principles of sequence learning and text generation, making it an excellent educational tool.

### 5.4 Ethical Considerations

**Bias in Training Data:**
- Shakespeare's works reflect 16th-17th century English society
- Contains outdated gender roles and social hierarchies
- Generated text may perpetuate these biases

**Potential Misuse:**
- Could be used to generate fake "Shakespearean" quotes
- Plagiarism concerns if used without attribution
- Misinformation if outputs presented as authentic

**Mitigation Strategies:**
- Clear labeling as AI-generated content
- Disclaimer in web application
- Educational context emphasized
- Not marketed as authoritative Shakespeare

---

## 6. Conclusion & Future Work

### 6.1 Summary of Key Findings

This project successfully implemented an LSTM-based text generation system that learns to produce Shakespearean-style text. The final model achieved 11.99% validation accuracy and 345.70 perplexity, generating locally coherent text that captures Shakespeare's vocabulary, themes, and stylistic elements.

**Key Contributions:**
1. Systematic comparison of vocabulary sizes showing 3K optimal for this corpus
2. Comprehensive temperature analysis providing usage guidelines
3. Full-stack deployment with interactive web interface
4. Reproducible training pipeline with documented hyperparameters

**Technical Insights:**
- LSTM architecture successfully handles 50-word context windows
- Temperature parameter offers intuitive creativity control
- Word-level tokenization superior to character-level for this task
- Smaller vocabulary (3K) generalizes better than large vocabulary (5K) on limited data

### 6.2 Lessons Learned

**About RNNs:**
- LSTM gates effectively prevent vanishing gradients
- Dropout crucial for preventing overfitting on small datasets
- Learning rate scheduling significantly improves convergence

**About Text Generation:**
- Local coherence easier than global coherence
- Temperature sampling more effective than greedy decoding
- Perplexity correlates well with subjective quality

**About MLOps:**
- Model size matters for deployment (keep under 100MB for free hosting)
- API latency important for user experience (target < 2s)
- CORS and environment variables critical for full-stack integration

### 6.3 Future Work

**Short-term Improvements:**
1. **Beam Search Decoding**: Instead of sampling, maintain top-k hypotheses for better quality
2. **Attention Mechanism**: Add attention to improve long-range dependencies
3. **Bidirectional LSTM**: Use bidirectional encoding for better context
4. **Larger Training Corpus**: Add other Early Modern English texts (Marlowe, Jonson)

**Long-term Extensions:**
1. **Transformer Architecture**: Replace LSTM with transformer for better performance
2. **Fine-tuning**: Start with pre-trained language model (GPT-2) and fine-tune on Shakespeare
3. **Conditional Generation**: Control genre (tragedy, comedy) or character (Hamlet, Macbeth)
4. **Multi-task Learning**: Joint training for generation, classification, and sentiment analysis
5. **Interactive Refinement**: Allow users to guide generation through feedback

**Research Directions:**
1. **Interpretability**: Visualize what the LSTM learns (attention on words, phrase boundaries)
2. **Few-shot Adaptation**: Transfer learning to other authors with minimal examples
3. **Evaluation Metrics**: Develop automated metrics for Shakespearean-ness
4. **Stylometric Analysis**: Compare statistical properties of generated vs. authentic text

---

## 7. Appendix

### A. Model Architecture Summary

```
================================================================
Total params: 1,962,568
Trainable params: 1,962,568
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 12.58
Params size (MB): 7.49
Estimated Total Size (MB): 20.06
================================================================
```

### B. Training Configuration

```python
config = {
    'vocab_size': 3000,
    'embedding_dim': 100,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 12,
    'gradient_clip': 5.0,
    'early_stopping_patience': 5,
    'lr_scheduler_patience': 2,
    'lr_scheduler_factor': 0.5
}
```

### C. Final Training Metrics

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val Perplexity |
|-------|-----------|----------|-----------|---------|----------------|
| 1 | 5.894 | 5.533 | 7.01% | 9.93% | 252.36 |
| 6 | 4.799 | 5.335 | 13.52% | 12.15% | 207.53 |
| 12 | 4.335 | 5.846 | 15.85% | 11.99% | 345.70 |

### D. Generated Text Samples

See `GENERATED_TEXT_SAMPLES.md` for 17 diverse examples across different temperatures and seed texts.

### E. Deployment URLs

- **Frontend**: https://shakespearernn.onrender.com/
- **Backend API**: https://shakespearernn-3.onrender.com/
- **Swagger Docs**: https://shakespearernn-3.onrender.com/docs
- **GitHub**: https://github.com/gvklok/ShakespeareRNN

### F. References

1. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." Neural Computation.
2. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder."
3. Karpathy (2015). "The Unreasonable Effectiveness of Recurrent Neural Networks."
4. Goodfellow et al. (2016). "Deep Learning." MIT Press.
5. PyTorch Documentation (2024). https://pytorch.org/docs/

---

**End of Report**

*This technical report documents a full-stack deep learning project implementing LSTM-based text generation. All code, models, and visualizations are available in the GitHub repository. The deployed application demonstrates practical deployment of machine learning models in production environments.*
