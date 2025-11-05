# Recurrent Neural Networks (RNN) - Text Generation Activity

**Building an Intelligent Text Generation System with Deep Learning**

---

## Table of Contents

1. [Introduction to Recurrent Neural Networks](#1-introduction-to-recurrent-neural-networks)
2. [RNN Architecture Deep Dive](#2-rnn-architecture-deep-dive)
3. [Activity Overview](#3-activity-overview)
4. [Step-by-Step Implementation](#4-step-by-step-implementation)
5. [Visualization Requirements](#5-visualization-requirements)
6. [Project Deliverables](#6-project-deliverables)
7. [Additional Resources](#7-additional-resources)

---

## 1. Introduction to Recurrent Neural Networks

### 1.1 What are RNNs?

**Recurrent Neural Networks (RNNs)** are a specialized class of neural networks designed to process **sequential data**. Unlike feedforward neural networks that assume inputs are independent, RNNs maintain an internal state (memory) that allows them to process sequences where the order and context matter.

### 1.2 Why Use RNNs?

RNNs excel at sequential tasks for several fundamental reasons:

#### **Memory Capability**
RNNs maintain a hidden state $h_t$ that acts as memory, carrying information from previous time steps to the current computation. This allows the network to "remember" context.

#### **Variable-Length Sequence Processing**
Unlike fixed-size input networks, RNNs can process sequences of arbitrary length:
- Sentences with different word counts
- Time series of varying duration
- Variable-length audio clips

#### **Parameter Sharing Across Time**
The same weight matrices are applied at each time step:
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

This means fewer parameters to learn compared to treating each time step independently.

#### **Temporal Dependency Capture**
RNNs can learn relationships between elements separated by many time steps, essential for:
- Understanding pronoun references in text
- Predicting future values based on historical patterns
- Recognizing patterns in sequential data

### 1.3 Common Applications

| Domain | Application | Example |
|--------|-------------|---------|
| **Natural Language Processing** | Machine Translation | English → Spanish |
| | Text Generation | Story/poetry creation |
| | Sentiment Analysis | Movie review classification |
| | Named Entity Recognition | Identifying people, places in text |
| **Time Series** | Stock Price Prediction | Financial forecasting |
| | Weather Forecasting | Temperature prediction |
| **Speech & Audio** | Speech Recognition | Voice-to-text |
| | Music Generation | Composing melodies |
| **Video Processing** | Activity Recognition | Detecting actions in videos |
| | Video Captioning | Describing video content |

---

## 2. RNN Architecture Deep Dive

### 2.1 Basic RNN Cell

The fundamental RNN unit processes one element of a sequence at each time step $t$:

```
      ┌─────────┐
x_t ──┤         │
      │  RNN    ├── h_t (output/hidden state)
h_t-1─┤  Cell   │
      └─────────┘
```

#### **Mathematical Formulation**

At each time step $t$, the RNN computes:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

Where:
- $h_t \in \mathbb{R}^{h}$ : Hidden state at time $t$ (memory)
- $x_t \in \mathbb{R}^{d}$ : Input at time $t$
- $y_t \in \mathbb{R}^{o}$ : Output at time $t$
- $W_{hh} \in \mathbb{R}^{h \times h}$ : Hidden-to-hidden weight matrix (recurrent weights)
- $W_{xh} \in \mathbb{R}^{h \times d}$ : Input-to-hidden weight matrix
- $W_{hy} \in \mathbb{R}^{o \times h}$ : Hidden-to-output weight matrix
- $b_h \in \mathbb{R}^{h}$ : Hidden bias vector
- $b_y \in \mathbb{R}^{o}$ : Output bias vector
- $\tanh$ : Hyperbolic tangent activation function

#### **Why Tanh?**

The $\tanh$ activation function is commonly used because:
- Output range: $[-1, 1]$ (centered around zero)
- Non-linear: Allows learning complex patterns
- Smooth gradient: Better than hard thresholds
- Zero-centered: Helps with optimization

### 2.2 Unfolding Through Time

To understand RNN computation, we "unfold" the recurrent network across time steps:

```
x_1      x_2      x_3      x_t
 │        │        │        │
 ▼        ▼        ▼        ▼
┌─┐  ┌─┐ ┌─┐  ┌─┐ ┌─┐  ┌─┐ ┌─┐
│h├─►│h├►│h├─►│h├►│h├─►│h├►│h│
└┬┘  └┬┘ └┬┘  └┬┘ └┬┘  └┬┘ └┬┘
 │    │   │    │   │    │   │
 ▼    ▼   ▼    ▼   ▼    ▼   ▼
y_1  y_2 y_3  ... y_t
```

Each box represents the same RNN cell with shared weights, applied at different time steps.

### 2.3 Forward Propagation in Detail

For a sequence $X = [x_1, x_2, ..., x_T]$:

1. **Initialize** hidden state: $h_0 = \vec{0}$ (or learned initialization)

2. **For each time step** $t = 1$ to $T$:
   - Compute pre-activation: $a_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h$
   - Apply activation: $h_t = \tanh(a_t)$
   - Compute output: $y_t = W_{hy} h_t + b_y$
   - Apply output activation (e.g., softmax for classification): $\hat{y}_t = \text{softmax}(y_t)$

3. **Return** sequence of outputs $[\hat{y}_1, ..., \hat{y}_T]$ and final hidden state $h_T$

### 2.4 Backpropagation Through Time (BPTT)

Training RNNs uses **Backpropagation Through Time (BPTT)**, which unfolds the network and applies standard backpropagation.

#### **Loss Function**

For sequence prediction with cross-entropy loss:

$$\mathcal{L} = -\sum_{t=1}^{T} \sum_{c=1}^{C} y_{t,c} \log(\hat{y}_{t,c})$$

Where:
- $C$ : Number of classes (vocabulary size for text)
- $y_{t,c}$ : True label (one-hot encoded)
- $\hat{y}_{t,c}$ : Predicted probability for class $c$ at time $t$

#### **Gradient Computation**

Gradients flow backward through time:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hh}}$$

At each time step, we compute:

$$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh} \cdot \text{diag}(1 - \tanh^2(a_t))$$

This chain rule application through many time steps leads to the **vanishing/exploding gradient problem**.

### 2.5 The Vanishing Gradient Problem

**Problem**: When $\frac{\partial h_t}{\partial h_{t-1}} < 1$, gradients shrink exponentially over many time steps:

$$\frac{\partial h_t}{\partial h_0} = \prod_{i=1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

If each factor $< 1$, the product approaches zero for large $t$.

**Consequence**: Network cannot learn long-term dependencies.

**Example**: In the sentence "The cat, which was sitting on the mat and grooming itself, **was** happy", the RNN must remember "cat" (singular) over many words to correctly use "was" instead of "were".

### 2.6 Long Short-Term Memory (LSTM)

**LSTM networks** solve the vanishing gradient problem using a gating mechanism.

#### **LSTM Architecture**

```
        ┌──────────────────────────────┐
        │    LSTM Cell at time t       │
        │                              │
   C_t-1├──────[×]─────[+]────────────┤─► C_t
        │       │       │              │
        │       f_t    [×]             │
        │              │               │
        │              ĩ_t             │
        │                              │
   h_t-1├──[σ]──[σ]───[tanh]──[σ]────┤─► h_t
        │  f_t  i_t     g_t    o_t    │
        │   │    │       │      │     │
   x_t ─┼───┴────┴───────┴──────┘     │
        └──────────────────────────────┘
```

#### **LSTM Equations**

At each time step $t$:

**Forget Gate** (what to remove from cell state):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** (what new information to add):
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State** (new information):
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update** (selective memory):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate** (what to output):
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State** (filtered cell state):
$$h_t = o_t \odot \tanh(C_t)$$

Where:
- $\sigma$ : Sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ (outputs 0-1)
- $\odot$ : Element-wise multiplication (Hadamard product)
- $C_t$ : Cell state (long-term memory)
- $h_t$ : Hidden state (short-term memory/output)

#### **Why LSTM Works**

1. **Additive Cell State Update**: $C_t = f_t \odot C_{t-1} + ...$
   - Addition (not multiplication) prevents gradient vanishing
   - Allows gradients to flow unchanged through time

2. **Gating Mechanism**:
   - Gates learn to preserve important information
   - Can maintain information over 100+ time steps

3. **Selective Memory**:
   - Forget gate removes irrelevant information
   - Input gate adds new relevant information
   - Output gate controls what to expose

### 2.7 Gated Recurrent Unit (GRU)

GRU is a simplified LSTM variant with fewer parameters:

**Update Gate**:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**Reset Gate**:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**Candidate Hidden State**:
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

**Final Hidden State**:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**GRU vs LSTM**:
- GRU: Fewer parameters, faster training, works well on smaller datasets
- LSTM: More expressive, better for complex tasks, more widely used

### 2.8 Bidirectional RNN

Processes sequences in both directions to capture future context:

```
Forward:  h⃗_1 → h⃗_2 → h⃗_3 → h⃗_t
             ↓      ↓      ↓      ↓
Input:      x_1    x_2    x_3    x_t
             ↓      ↓      ↓      ↓
Backward: h⃖_1 ← h⃖_2 ← h⃖_3 ← h⃖_t

Output:   y_1 = f([h⃗_1, h⃖_1])
```

At each time step: $h_t = [h⃗_t, h⃖_t]$ (concatenation)

**Use Cases**: Tasks where future context helps (text classification, named entity recognition)

### 2.9 Many-to-Many RNN Architectures

For text generation, we use a **Many-to-Many** architecture:

**Approach 1: Teacher Forcing (Training)**
```
Input:     <START> "the"  "cat"  "sat"
            ↓      ↓      ↓      ↓
RNN:       h_1 →  h_2 →  h_3 →  h_4
            ↓      ↓      ↓      ↓
Output:    "the"  "cat"  "sat"  "down"
```

**Approach 2: Auto-regressive (Generation)**
```
Input:     <START> ──→ predict "the" ──→ predict "cat" ──→ ...
```

Use previous prediction as next input.

### 2.10 Key Architectural Decisions

When designing an RNN, consider:

| Component | Options | Trade-offs |
|-----------|---------|------------|
| **Cell Type** | Simple RNN, LSTM, GRU | LSTM/GRU for long sequences, Simple RNN for short |
| **Layers** | 1-4 layers | More layers = more capacity but slower, risk overfitting |
| **Hidden Units** | 128, 256, 512, 1024 | More units = more capacity but more computation |
| **Bidirectional** | Yes/No | Yes for classification, No for generation |
| **Dropout** | 0.2 - 0.5 | Prevents overfitting, essential for deep RNNs |
| **Embedding Dim** | 50, 100, 200, 300 | Larger for larger vocabularies |

---

## 3. Activity Overview

### 3.1 Project Goal

Build a **production-ready text generation web application** that:
- Trains an LSTM model on a text corpus
- Generates coherent, stylistically similar text
- Provides an interactive web interface (React frontend)
- Exposes an API for text generation (FastAPI backend)
- Visualizes the training process and model architecture
- Deploys to a free cloud platform

### 3.2 Learning Objectives

By completing this activity, you will:

1. **Understand RNN/LSTM architecture** at a deep mathematical and conceptual level
2. **Implement text preprocessing** (tokenization, encoding, sequence generation)
3. **Build and train** deep learning models using PyTorch
4. **Develop full-stack applications** with React and FastAPI
5. **Visualize model training** and architecture
6. **Deploy ML applications** to production cloud environments
7. **Experiment with hyperparameters** and analyze their effects

### 3.3 Technology Stack

- **Deep Learning**: PyTorch 2.x (modern, flexible deep learning framework)
- **Backend**: FastAPI (modern, async Python framework)
- **Frontend**: React with TypeScript
- **Visualization**: Matplotlib, TensorBoard, Plotly
- **Deployment**: Render, Railway, or Hugging Face Spaces
- **Version Control**: Git + GitHub

---

## 4. Step-by-Step Implementation

### Step 1: Environment Setup

#### 4.1.1 Project Structure

Create the following directory structure:

```
rnn-text-generator/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic models
│   │   ├── text_generator.py   # RNN model class
│   │   └── train.py             # Training script
│   ├── data/
│   │   └── training_text.txt    # Your training corpus
│   ├── saved_models/
│   │   ├── model.h5             # Trained model
│   │   └── tokenizer.pkl        # Tokenizer
│   ├── visualizations/
│   │   └── (training plots)
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TextGenerator.tsx
│   │   │   ├── ModelVisualizer.tsx
│   │   │   └── TrainingMetrics.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

#### 4.1.2 Backend Dependencies

Create `backend/requirements.txt`:

```txt
# Deep Learning
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3

# API Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Utilities
pydantic==2.5.0
python-dotenv==1.0.0
tqdm==4.66.1

# Visualization
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0
tensorboard==2.15.1

# Model Persistence (built into torch)
# No additional libraries needed
```

#### 4.1.3 Frontend Setup

Initialize React with TypeScript:

```bash
cd frontend
npx create-react-app . --template typescript
npm install axios recharts @types/recharts
npm install -D tailwindcss postcss autoprefixer
```

---

### Step 2: Obtaining Training Data

#### 4.2.1 Where Does Training Data Come From?

Your RNN needs a large text corpus to learn from. Here are practical methods to obtain training data:

#### Option 1: Download from Project Gutenberg (Recommended for Beginners)

**Project Gutenberg** offers free classic literature in plain text format.

**Step-by-step:**

1. Visit [www.gutenberg.org](https://www.gutenberg.org)
2. Search for a book (e.g., "Alice in Wonderland", "Pride and Prejudice", "Sherlock Holmes")
3. Click on the book title
4. Look for "Plain Text UTF-8" format
5. Download the `.txt` file
6. Save it as `backend/data/training_text.txt`

**Using Python to download:**

```python
import urllib.request

# Download Alice in Wonderland
url = "https://www.gutenberg.org/files/11/11-0.txt"
output_path = "backend/data/training_text.txt"

urllib.request.urlretrieve(url, output_path)
print(f"Downloaded to {output_path}")
```

**Clean the file** (remove Project Gutenberg headers/footers):

```python
def clean_gutenberg_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove header (before "*** START OF")
    start_marker = "*** START OF"
    if start_marker in text:
        text = text.split(start_marker)[1]

    # Remove footer (after "*** END OF")
    end_marker = "*** END OF"
    if end_marker in text:
        text = text.split(end_marker)[0]

    # Save cleaned text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text.strip())

    print(f"Cleaned text saved to {output_file}")

# Usage
clean_gutenberg_text("backend/data/raw_download.txt", "backend/data/training_text.txt")
```

**Recommended Books:**
- Alice in Wonderland: `https://www.gutenberg.org/files/11/11-0.txt`
- Sherlock Holmes: `https://www.gutenberg.org/files/1661/1661-0.txt`
- Pride and Prejudice: `https://www.gutenberg.org/files/1342/1342-0.txt`
- Complete Works of Shakespeare: `https://www.gutenberg.org/files/100/100-0.txt` (very large!)

#### Option 2: Scrape Wikipedia Articles

**For a specific topic:**

```python
import requests
from bs4 import BeautifulSoup

def download_wikipedia_article(title):
    """Download text from a Wikipedia article."""
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract paragraphs
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])

    return text

# Download multiple related articles
topics = ["Machine_learning", "Artificial_intelligence", "Neural_network",
          "Deep_learning", "Natural_language_processing"]

all_text = ""
for topic in topics:
    print(f"Downloading {topic}...")
    all_text += download_wikipedia_article(topic) + "\n\n"

# Save
with open("backend/data/training_text.txt", 'w', encoding='utf-8') as f:
    f.write(all_text)

print(f"Total characters: {len(all_text)}")
```

**Install required libraries:**
```bash
pip install beautifulsoup4 requests
```

#### Option 3: Use Existing Datasets

**TensorFlow Datasets:**

```python
import tensorflow_datasets as tfds

# Download IMDB movie reviews
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Combine all text
all_text = []
for text, label in dataset['train']:
    all_text.append(text.numpy().decode('utf-8'))

# Save to file
combined_text = ' '.join(all_text)
with open("backend/data/training_text.txt", 'w', encoding='utf-8') as f:
    f.write(combined_text)
```

#### Option 4: Use Song Lyrics

**From Kaggle:**

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Search for "song lyrics" or "rap lyrics"
3. Download CSV file
4. Extract text column:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('downloaded_lyrics.csv')

# Combine all lyrics
all_lyrics = ' '.join(df['lyrics'].dropna().tolist())

# Save
with open("backend/data/training_text.txt", 'w', encoding='utf-8') as f:
    f.write(all_lyrics)

print(f"Total songs: {len(df)}")
print(f"Total characters: {len(all_lyrics)}")
```

#### Option 5: Reddit Comments

**Using PRAW (Python Reddit API Wrapper):**

```python
import praw

# Create Reddit instance (you'll need API credentials from reddit.com/prefs/apps)
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='text_collector'
)

# Collect comments from a subreddit
subreddit = reddit.subreddit('jokes')
comments_text = []

for submission in subreddit.hot(limit=100):
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        comments_text.append(comment.body)

# Save
all_text = '\n'.join(comments_text)
with open("backend/data/training_text.txt", 'w', encoding='utf-8') as f:
    f.write(all_text)
```

#### Option 6: Create Your Own Dataset

**Combine multiple sources:**

```python
# Read multiple files
sources = []

# Add a book
with open("book1.txt", 'r') as f:
    sources.append(f.read())

# Add articles
with open("articles.txt", 'r') as f:
    sources.append(f.read())

# Add custom text
sources.append("""
Your own creative writing or text here.
Can be anything you want the model to learn from.
""")

# Combine and save
final_text = '\n\n'.join(sources)
with open("backend/data/training_text.txt", 'w', encoding='utf-8') as f:
    f.write(final_text)
```

#### 4.2.2 Data Requirements & Validation

**Minimum Requirements:**
- **Size**: At least 100KB (~20,000 words)
- **Format**: Plain text (.txt) with UTF-8 encoding
- **Location**: `backend/data/training_text.txt`

**Check your data:**

```python
import os

def validate_training_data(file_path):
    """Check if training data is suitable."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Statistics
    num_chars = len(text)
    num_words = len(text.split())
    num_lines = len(text.split('\n'))
    unique_words = len(set(text.lower().split()))

    print(f"✓ File exists: {file_path}")
    print(f"✓ Size: {num_chars / 1024:.2f} KB")
    print(f"✓ Characters: {num_chars:,}")
    print(f"✓ Words: {num_words:,}")
    print(f"✓ Lines: {num_lines:,}")
    print(f"✓ Unique words: {unique_words:,}")

    # Validate
    if num_chars < 100_000:
        print(f"⚠️  Warning: Text is small ({num_chars/1024:.1f}KB). Recommend >100KB")

    if unique_words < 500:
        print(f"⚠️  Warning: Low vocabulary ({unique_words} words). Recommend >500")

    return True

# Usage
validate_training_data("backend/data/training_text.txt")
```

**Example Output:**
```
✓ File exists: backend/data/training_text.txt
✓ Size: 187.45 KB
✓ Characters: 192,012
✓ Words: 35,428
✓ Lines: 3,245
✓ Unique words: 6,832
```

#### 4.2.3 Quick Start Script

Create `backend/scripts/download_data.py`:

```python
#!/usr/bin/env python3
"""
Quick script to download and prepare training data.
"""

import urllib.request
import os

def download_sample_data():
    """Download Alice in Wonderland as sample data."""

    # Create data directory
    os.makedirs("data", exist_ok=True)

    print("Downloading Alice in Wonderland from Project Gutenberg...")
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    output_path = "data/training_text.txt"

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")

        # Clean the file
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Remove Project Gutenberg header/footer
        if "*** START OF" in text:
            text = text.split("*** START OF")[1]
        if "*** END OF" in text:
            text = text.split("*** END OF")[0]

        # Save cleaned version
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())

        # Show stats
        print(f"✓ Cleaned text")
        print(f"  - Characters: {len(text):,}")
        print(f"  - Words: {len(text.split()):,}")
        print(f"  - Size: {len(text)/1024:.2f} KB")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    download_sample_data()
```

**Run it:**
```bash
cd backend
python scripts/download_data.py
```

**Your directory should now look like:**
```
backend/
├── data/
│   └── training_text.txt  ← Your training data is here!
├── scripts/
│   └── download_data.py
└── app/
    └── ...
```

---

### Step 3: Data Preprocessing Pipeline

#### 4.3.1 Text Generator Class

Create `backend/app/text_generator.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class TextGenerator:
    """
    Advanced RNN-based text generator with LSTM architecture.

    This class handles:
    - Text preprocessing and tokenization
    - Sequence generation for training
    - LSTM model construction
    - Training with visualization
    - Text generation with temperature sampling
    """

    def __init__(
        self,
        sequence_length: int = 50,
        embedding_dim: int = 100,
        lstm_units: int = 150,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the text generator.

        Args:
            sequence_length: Number of words to consider for context
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of units in each LSTM layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = 0
        self.max_sequence_len = 0
        self.history = None

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize input text.

        Steps:
        1. Convert to lowercase
        2. Remove special characters (keep punctuation)
        3. Normalize whitespace
        4. Remove extra newlines

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'\-]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def prepare_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Convert text to training sequences.

        Process:
        1. Tokenize text into words
        2. Build vocabulary
        3. Create sliding window sequences
        4. Encode as integer sequences
        5. Separate into X (input) and y (target)

        Args:
            text: Preprocessed text

        Returns:
            Tuple of (X, y, max_sequence_len)
            - X: Input sequences (context words)
            - y: Target words (one-hot encoded)
            - max_sequence_len: Length of longest sequence
        """
        # Preprocess
        text = self.preprocess_text(text)

        # Tokenize text
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        print(f"Vocabulary size: {self.vocab_size}")

        # Create input sequences using sliding window
        input_sequences = []
        words = text.split()

        for i in range(self.sequence_length, len(words)):
            # Take sequence_length + 1 words
            # First sequence_length words = input
            # Last word = target
            seq = words[i - self.sequence_length : i + 1]
            input_sequences.append(seq)

        print(f"Total sequences: {len(input_sequences)}")

        # Convert words to integer sequences
        token_sequences = self.tokenizer.texts_to_sequences(input_sequences)

        # Pad sequences to same length
        self.max_sequence_len = max([len(seq) for seq in token_sequences])
        padded_sequences = pad_sequences(
            token_sequences,
            maxlen=self.max_sequence_len,
            padding='pre'
        )

        # Split into inputs and labels
        X = padded_sequences[:, :-1]  # All but last word
        y = padded_sequences[:, -1]   # Last word

        # Convert y to one-hot encoding
        y = keras.utils.to_categorical(y, num_classes=self.vocab_size)

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y, self.max_sequence_len

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Architecture:
        1. Embedding layer (word → dense vector)
        2. Multiple LSTM layers with dropout
        3. Dense output layer with softmax

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name="Text_Generator_LSTM")

        # Embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_len - 1,
            name="Embedding"
        ))

        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)
            model.add(layers.LSTM(
                units=self.lstm_units,
                return_sequences=return_sequences,
                name=f"LSTM_{i+1}"
            ))
            model.add(layers.Dropout(
                rate=self.dropout_rate,
                name=f"Dropout_{i+1}"
            ))

        # Output layer
        model.add(layers.Dense(
            units=self.vocab_size,
            activation='softmax',
            name="Output"
        ))

        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        validation_split: float = 0.1,
        save_path: str = "saved_models"
    ) -> Dict:
        """
        Train the LSTM model with visualization.

        Args:
            X: Input sequences
            y: Target words (one-hot)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            save_path: Directory to save checkpoints

        Returns:
            Dictionary with training history
        """
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"{save_path}/model_best.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f"{save_path}/logs",
                histogram_freq=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def generate_text(
        self,
        seed_text: str,
        num_words: int = 50,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using trained model.

        Temperature controls randomness:
        - Low (0.5): More predictable, coherent
        - Medium (1.0): Balanced
        - High (1.5-2.0): More creative, random

        Args:
            seed_text: Starting text
            num_words: Number of words to generate
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        generated_text = seed_text.lower()

        for _ in range(num_words):
            # Tokenize current text
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]

            # Take last sequence_length tokens
            token_list = token_list[-(self.sequence_length):]

            # Pad to model input size
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_len - 1,
                padding='pre'
            )

            # Predict next word probabilities
            predicted_probs = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature
            predicted_probs = np.log(predicted_probs + 1e-10) / temperature
            predicted_probs = np.exp(predicted_probs)
            predicted_probs = predicted_probs / np.sum(predicted_probs)

            # Sample from distribution
            predicted_index = np.random.choice(
                len(predicted_probs),
                p=predicted_probs
            )

            # Convert index to word
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    generated_text += " " + word
                    break

        return generated_text

    def visualize_architecture(self, save_path: str = "visualizations"):
        """Generate model architecture visualization."""
        keras.utils.plot_model(
            self.model,
            to_file=f"{save_path}/model_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )

    def plot_training_history(self, save_path: str = "visualizations"):
        """
        Plot training and validation metrics.

        Creates two subplots:
        1. Loss over epochs
        2. Accuracy over epochs
        """
        if self.history is None:
            raise ValueError("No training history available!")

        history_dict = self.history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(history_dict['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: str, tokenizer_path: str):
        """Save model and tokenizer."""
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_len': self.max_sequence_len
        }

        with open(model_path.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path: str, tokenizer_path: str):
        """Load saved model and tokenizer."""
        self.model = keras.models.load_model(model_path)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Load configuration
        config_path = model_path.replace('.h5', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = config['vocab_size']
        self.max_sequence_len = config['max_sequence_len']
```

#### 4.2.2 Training Script

Create `backend/app/train.py`:

```python
from text_generator import TextGenerator
import os
import sys

def main():
    """Main training pipeline."""

    # Configuration
    DATA_PATH = "data/training_text.txt"
    MODEL_DIR = "saved_models"
    VIZ_DIR = "visualizations"

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load training data
    print("Loading training data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")
    print(f"Unique words: {len(set(text.split()))}")

    # Initialize generator
    print("\nInitializing text generator...")
    generator = TextGenerator(
        sequence_length=50,
        embedding_dim=100,
        lstm_units=150,
        num_layers=2,
        dropout_rate=0.2
    )

    # Prepare sequences
    print("\nPreparing training sequences...")
    X, y, max_seq_len = generator.prepare_sequences(text)

    # Build model
    print("\nBuilding model...")
    model = generator.build_model()
    model.summary()

    # Visualize architecture
    print("\nGenerating architecture visualization...")
    generator.visualize_architecture(save_path=VIZ_DIR)

    # Train model
    print("\nTraining model...")
    history = generator.train(
        X, y,
        epochs=100,
        batch_size=128,
        validation_split=0.1,
        save_path=MODEL_DIR
    )

    # Plot training history
    print("\nGenerating training plots...")
    generator.plot_training_history(save_path=VIZ_DIR)

    # Save final model
    print("\nSaving model...")
    generator.save_model(
        f"{MODEL_DIR}/model.h5",
        f"{MODEL_DIR}/tokenizer.pkl"
    )

    # Test generation
    print("\n" + "="*50)
    print("Testing text generation...")
    print("="*50)

    seed_text = " ".join(text.split()[:10])
    print(f"\nSeed text: '{seed_text}'")

    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temp} ---")
        generated = generator.generate_text(seed_text, num_words=50, temperature=temp)
        print(generated)

    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
```

---

### Step 3: FastAPI Backend

#### 4.3.1 Pydantic Models

Create `backend/app/models.py`:

```python
from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    seed_text: str = Field(..., min_length=1, description="Starting text")
    num_words: int = Field(50, ge=10, le=200, description="Number of words to generate")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    seed_text: str
    num_words: int
    temperature: float

class ModelInfo(BaseModel):
    """Model information."""
    vocab_size: int
    sequence_length: int
    embedding_dim: int
    lstm_units: int
    num_layers: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
```

#### 4.3.2 FastAPI Application

Create `backend/app/main.py`:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse
)
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="RNN Text Generator API",
    description="Generate text using LSTM neural networks",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator = None
MODEL_PATH = "saved_models/model.h5"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global generator

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            generator = TextGenerator()
            generator.load_model(MODEL_PATH, TOKENIZER_PATH)
            print("✓ Model loaded successfully")
        else:
            print("⚠ Model files not found. Train the model first.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        vocab_size=generator.vocab_size,
        sequence_length=generator.sequence_length,
        embedding_dim=generator.embedding_dim,
        lstm_units=generator.lstm_units,
        num_layers=generator.num_layers
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from seed."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature
        )

        return GenerateResponse(
            generated_text=generated,
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/architecture")
async def get_architecture():
    """Get model architecture diagram."""
    path = "visualizations/model_architecture.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found")
    return FileResponse(path)

@app.get("/visualizations/training")
async def get_training_history():
    """Get training history plot."""
    path = "visualizations/training_history.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training history not found")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Step 4: React Frontend

#### 4.4.1 API Service

Create `frontend/src/services/api.ts`:

```typescript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GenerateRequest {
  seed_text: string;
  num_words: number;
  temperature: number;
}

export interface GenerateResponse {
  generated_text: string;
  seed_text: string;
  num_words: number;
  temperature: number;
}

export interface ModelInfo {
  vocab_size: number;
  sequence_length: number;
  embedding_dim: number;
  lstm_units: number;
  num_layers: number;
}

export const generateText = async (request: GenerateRequest): Promise<GenerateResponse> => {
  const response = await api.post<GenerateResponse>('/generate', request);
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get<ModelInfo>('/model/info');
  return response.data;
};

export const getArchitectureImage = (): string => {
  return `${API_BASE_URL}/visualizations/architecture`;
};

export const getTrainingHistoryImage = (): string => {
  return `${API_BASE_URL}/visualizations/training`;
};
```

#### 4.4.2 Text Generator Component

Create `frontend/src/components/TextGenerator.tsx`:

```typescript
import React, { useState } from 'react';
import { generateText, GenerateResponse } from '../services/api';

const TextGenerator: React.FC = () => {
  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter some seed text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await generateText({
        seed_text: seedText,
        num_words: numWords,
        temperature: temperature,
      });
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-3xl font-bold mb-6 text-indigo-600">
          Generate Text
        </h2>

        {/* Seed Text Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Seed Text (starting words):
          </label>
          <textarea
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            rows={4}
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            placeholder="Enter some text to start with..."
          />
        </div>

        {/* Number of Words */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Words: {numWords}
          </label>
          <input
            type="range"
            min="10"
            max="200"
            value={numWords}
            onChange={(e) => setNumWords(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Temperature Slider */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Creativity (Temperature): {temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full"
          />
          <p className="text-sm text-gray-500 mt-1">
            Lower = more predictable, Higher = more creative
          </p>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition-colors"
        >
          {loading ? 'Generating...' : 'Generate Text'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {/* Generated Text */}
        {result && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
            <h3 className="text-lg font-semibold mb-2">Generated Text:</h3>
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
              {result.generated_text}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextGenerator;
```

#### 4.4.3 Model Visualizer Component

Create `frontend/src/components/ModelVisualizer.tsx`:

```typescript
import React, { useEffect, useState } from 'react';
import { getModelInfo, getArchitectureImage, getTrainingHistoryImage, ModelInfo } from '../services/api';

const ModelVisualizer: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await getModelInfo();
        setModelInfo(info);
      } catch (err) {
        console.error('Failed to load model info');
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return <div className="text-center p-8">Loading model information...</div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-3xl font-bold mb-6 text-indigo-600">
        Model Architecture & Training
      </h2>

      {/* Model Info Cards */}
      {modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Vocabulary Size</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.vocab_size.toLocaleString()}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">LSTM Units</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.lstm_units}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Number of Layers</h3>
            <p className="text-3xl font-bold text-indigo-600">{modelInfo.num_layers}</p>
          </div>
        </div>
      )}

      {/* Architecture Diagram */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
        <h3 className="text-xl font-semibold mb-4">Model Architecture</h3>
        <img
          src={getArchitectureImage()}
          alt="Model Architecture"
          className="w-full rounded-lg"
        />
      </div>

      {/* Training History */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Training History</h3>
        <img
          src={getTrainingHistoryImage()}
          alt="Training History"
          className="w-full rounded-lg"
        />
      </div>
    </div>
  );
};

export default ModelVisualizer;
```

#### 4.4.4 Main App Component

Create `frontend/src/App.tsx`:

```typescript
import React, { useState } from 'react';
import TextGenerator from './components/TextGenerator';
import ModelVisualizer from './components/ModelVisualizer';

function App() {
  const [activeTab, setActiveTab] = useState<'generate' | 'visualize'>('generate');

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="container mx-auto py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">
            RNN Text Generator
          </h1>
          <p className="text-xl text-indigo-100">
            Powered by LSTM Neural Networks
          </p>
        </header>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-lg shadow-lg p-1 inline-flex">
            <button
              onClick={() => setActiveTab('generate')}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                activeTab === 'generate'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
            >
              Generate Text
            </button>
            <button
              onClick={() => setActiveTab('visualize')}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                activeTab === 'visualize'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
            >
              Model Info
            </button>
          </div>
        </div>

        {/* Content */}
        {activeTab === 'generate' ? <TextGenerator /> : <ModelVisualizer />}

        {/* Footer */}
        <footer className="text-center mt-12 text-white">
          <p>AIT-204: Recurrent Neural Networks Activity</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
```

---

### Step 5: Local Testing

#### 4.5.1 Run Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model
python app/train.py

# Run API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 4.5.2 Run Frontend

```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000`

---

### Step 6: Cloud Deployment

#### 4.6.1 Option A: Render

**Backend Deployment:**

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: rnn-backend
    env: python
    buildCommand: "pip install -r backend/requirements.txt"
    startCommand: "cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

2. Push to GitHub
3. Connect to Render
4. Deploy

**Frontend Deployment:**

1. Build React app: `npm run build`
2. Deploy build folder to Render Static Site

#### 4.6.2 Option B: Railway

1. Push code to GitHub
2. Create new project on Railway
3. Add backend service (auto-detects Python)
4. Add frontend service (auto-detects Node.js)
5. Set environment variables
6. Deploy

#### 4.6.3 Option C: Docker Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/saved_models:/app/saved_models
      - ./backend/visualizations:/app/visualizations
    environment:
      - PORT=8000

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
```

Deploy to any platform supporting Docker (Railway, Render, Fly.io).

---

## 5. Visualization Requirements

Your project must include the following visualizations:

### 5.1 Model Architecture Diagram

**Required Elements:**
- Layer names and types (Embedding, LSTM, Dense)
- Input/output shapes for each layer
- Parameter counts
- Connections between layers

**Tool:** `keras.utils.plot_model()` or TensorBoard

**Example Output:**
```
Input (None, 49)
    ↓
Embedding (None, 49, 100) - 500,000 params
    ↓
LSTM_1 (None, 49, 150) - 150,600 params
    ↓
Dropout (None, 49, 150)
    ↓
LSTM_2 (None, 150) - 180,600 params
    ↓
Dropout (None, 150)
    ↓
Dense (None, 5000) - 755,000 params
    ↓
Softmax Output
```

### 5.2 Training Metrics

**Required Plots:**

1. **Loss Curves** (Training vs Validation)
   - X-axis: Epochs
   - Y-axis: Loss value
   - Both training and validation on same plot

2. **Accuracy Curves** (Training vs Validation)
   - X-axis: Epochs
   - Y-axis: Accuracy (%)
   - Both training and validation on same plot

3. **Learning Rate Schedule** (if using ReduceLROnPlateau)
   - Show how learning rate changes over time

**Tools:** Matplotlib, Seaborn, or TensorBoard

### 5.3 Text Generation Analysis

**Required Visualizations:**

1. **Temperature Comparison**
   - Generate same seed text with temperatures [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
   - Display side-by-side to show creativity differences

2. **Word Frequency Distribution**
   - Bar chart of most common words in generated vs training text
   - Shows if model learned vocabulary distribution

3. **Sequence Length Impact**
   - Compare generation quality with different context lengths
   - Table or chart showing coherence scores

### 5.4 Attention/Hidden State Visualization (Bonus)

For advanced students:

- Visualize hidden state activations over time
- Heatmap showing which words activate which neurons
- Attention weights (if using attention mechanism)

### 5.5 Interactive Dashboard

Your React frontend should display:
- Real-time generation
- Live parameter adjustment (temperature, length)
- Side-by-side architecture and metrics
- Model statistics (vocab size, layers, parameters)

---

## 6. Project Deliverables

### Deliverable 1: Complete Codebase (30%)

**Requirements:**
- Full source code for backend (FastAPI + TensorFlow)
- Full source code for frontend (React + TypeScript)
- Proper project structure as outlined
- Clean, well-commented code following PEP 8 (Python) and ESLint (TypeScript)
- `requirements.txt` and `package.json` with all dependencies
- `.gitignore` file (exclude `venv/`, `node_modules/`, `*.pyc`, etc.)

**GitHub Repository Must Include:**
- README.md with:
  - Project description
  - Setup instructions
  - Usage guide
  - API documentation
  - Deployment URL
- Code organized in logical directories
- Commit history showing development progress

### Deliverable 2: Trained Model & Visualizations (25%)

**Model Files:**
- `model.h5` (or `.keras` format)
- `tokenizer.pkl`
- `model_config.json`

**Required Visualizations:**
1. Model architecture diagram (PNG/PDF)
2. Training history plots (loss and accuracy)
3. Temperature comparison examples (at least 5 temperatures)
4. Word frequency distribution chart
5. Any additional analysis visualizations

**Submission Format:**
- All visualizations in `visualizations/` directory
- High resolution (300 DPI minimum)
- Clear labels and titles
- Include in web interface

### Deliverable 3: Deployed Application (25%)

**Requirements:**
- Backend API deployed and accessible
- Frontend web app deployed and accessible
- Both connected and functional
- Provide deployment URLs in README

**Deployment Checklist:**
- [ ] Backend responds to health check
- [ ] `/generate` endpoint works
- [ ] `/model/info` returns correct data
- [ ] Visualizations are accessible
- [ ] Frontend loads without errors
- [ ] Can generate text through web interface
- [ ] API calls succeed from frontend to backend
- [ ] CORS configured correctly
- [ ] Environment variables set properly

**Accepted Platforms:**
- Render (recommended)
- Railway
- Fly.io
- Hugging Face Spaces
- Vercel (frontend) + Render (backend)
- Any free-tier cloud platform

### Deliverable 4: Technical Report (20%)

**Format:** PDF, 4-6 pages, 12pt font, 1-inch margins

**Required Sections:**

#### 1. Introduction (0.5 pages)
- Brief overview of RNNs and LSTMs
- Explanation of text generation task
- Project objectives

#### 2. RNN Architecture Analysis (1 page)
- Detailed explanation of your model architecture
- Why you chose specific hyperparameters
- Mathematical formulation of your LSTM layers
- Comparison to alternatives (simple RNN, GRU, Transformers)

#### 3. Implementation Details (1.5 pages)
- Data preprocessing pipeline
- Tokenization strategy (word-level vs character-level)
- Training configuration:
  - Batch size and why
  - Learning rate and optimizer
  - Loss function and metrics
  - Regularization techniques (dropout, early stopping)
- API design decisions
- Frontend architecture choices

#### 4. Experiments & Results (2 pages)

**Minimum 3 Experiments Required:**

**Suggested Experiments:**
1. **Hyperparameter Tuning**
   - Vary LSTM units: [100, 150, 200, 256]
   - Vary layers: [1, 2, 3]
   - Vary sequence length: [25, 50, 100]
   - Document impact on loss, accuracy, training time

2. **Temperature Analysis**
   - Generate text with temperatures [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
   - Analyze coherence, creativity, repetition
   - Include examples

3. **Training Data Size Impact**
   - Train on 25%, 50%, 75%, 100% of data
   - Compare generation quality
   - Plot learning curves

4. **Architecture Comparison**
   - Simple RNN vs LSTM vs GRU
   - Same dataset and hyperparameters
   - Compare performance and speed

**For Each Experiment:**
- State hypothesis
- Describe methodology
- Present results (tables, charts)
- Analyze findings
- Draw conclusions

#### 5. Analysis & Discussion (1 page)
- Evaluate generated text quality:
  - Grammar and coherence
  - Style similarity to training data
  - Common errors or patterns
- Discuss challenges encountered:
  - Vanishing gradients
  - Overfitting
  - Training time
  - Deployment issues
- Limitations of your approach
- Comparison to state-of-the-art (GPT, BERT, etc.)
- Ethical considerations (bias, misuse)

#### 6. Conclusion & Future Work (0.5 pages)
- Summary of key findings
- What you learned about RNNs
- Suggestions for improvement:
  - Attention mechanisms
  - Beam search
  - Fine-tuning techniques
  - Larger models or datasets

#### 7. Appendix (Not counted in page limit)
- Generated text examples (10+ samples)
- Full model architecture printout
- Training logs
- API documentation
- Deployment screenshots

### Deliverable 5: Generated Text Samples (10%)

**Requirements:**
- Minimum 15 diverse text samples
- Must demonstrate:
  - Different seed texts
  - Different temperatures (0.5, 1.0, 1.5, 2.0)
  - Different generation lengths (25, 50, 100, 150 words)
  - Different training datasets (if you experiment with multiple)

**Presentation:**
- Create a document (PDF or HTML) with:
  - Seed text highlighted
  - Generated continuation
  - Parameters used (temperature, length)
  - Your evaluation (coherence, quality, issues)

**Include Both:**
- Successful generations (coherent, interesting)
- Problematic outputs (nonsense, repetition) with analysis

### Deliverable 6: Presentation (10%)

**Format:** 5-minute video or live presentation

**Required Slides:**
1. **Title & Team** (if group project)
2. **RNN Architecture Explanation** with diagrams
3. **Demo** of deployed application
4. **Key Experiments** and findings
5. **Interesting Generated Examples**
6. **Challenges & Solutions**
7. **Conclusion & Takeaways**

**Tips:**
- Show live demo of web app
- Use visualizations from your project
- Explain technical concepts clearly
- Discuss what you learned

---

## 7. Additional Resources

### 7.1 Recommended Reading

**Foundational Papers:**
1. [Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"](http://www.bioinf.jku.at/publications/older/2604.pdf)
2. [Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078) (introduces GRU)
3. [Karpathy (2015) - "The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

**Tutorials:**
- [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch Text Generation Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)

### 7.2 Datasets

**Text Corpora:**
- [Project Gutenberg](https://www.gutenberg.org/) - Classic literature
- [Kaggle Text Datasets](https://www.kaggle.com/datasets?tags=13204-text+data)
- [Reddit Comments Dataset](https://www.reddit.com/r/datasets/)
- [OpenWebText](https://github.com/jcpeterson/openwebtext) - Web text
- [Wikipedia Dumps](https://dumps.wikimedia.org/)

**Recommended Sizes:**
- Minimum: 100KB (~20,000 words)
- Good: 1-5MB (~200,000-1M words)
- Excellent: 10MB+ (1M+ words)

### 7.3 Tools & Libraries

**Python:**
- PyTorch - Deep learning framework
- FastAPI - Web framework
- Uvicorn - ASGI server
- Pydantic - Data validation
- Matplotlib/Seaborn - Visualization
- tqdm - Progress bars

**JavaScript/TypeScript:**
- React - Frontend framework
- Axios - HTTP client
- Recharts - Charting library
- TailwindCSS - Styling

**Deployment:**
- Docker - Containerization
- GitHub Actions - CI/CD
- Render/Railway/Fly.io - Hosting

### 7.4 Troubleshooting Guide

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **Model generates gibberish** | Insufficient training, poor data | Train longer (100+ epochs), use larger/cleaner dataset |
| **Repetitive output** | Overfitting, low temperature | Increase dropout, use higher temperature, more diverse data |
| **Loss not decreasing** | Learning rate too high/low, bad architecture | Adjust learning rate (0.001 → 0.0001), try different optimizer |
| **Out of memory** | Batch size too large, model too big | Reduce batch size (128 → 64 → 32), reduce LSTM units |
| **Slow training** | Large model, CPU training | Reduce model size, use GPU (Google Colab), reduce sequence length |
| **CORS errors** | Frontend can't reach backend | Enable CORS in FastAPI, check URLs, use proxy |
| **Deployment fails** | Missing dependencies, large files | Check requirements.txt, exclude large files from git, use .gitignore |

### 7.5 Evaluation Metrics

**Quantitative:**
- **Perplexity**: $PP(W) = P(w_1, w_2, ..., w_N)^{-1/N}$ (lower is better)
- **BLEU Score**: For comparing to reference text (0-1, higher is better)
- **Diversity**: Unique n-grams / Total n-grams

**Qualitative:**
- Grammar correctness (1-5 scale)
- Coherence (1-5 scale)
- Style similarity to training data (1-5 scale)
- Creativity/interestingness (1-5 scale)

### 7.6 Suggested Timeline

| Week | Tasks | Milestone |
|------|-------|-----------|
| **Week 1** | • Study RNN theory<br>• Set up development environment<br>• Collect and prepare training data<br>• Implement preprocessing pipeline | Dataset ready, basic code structure |
| **Week 2** | • Build LSTM model<br>• Implement training script<br>• Train initial model<br>• Generate first text samples | Working model that generates text |
| **Week 3** | • Build FastAPI backend<br>• Create React frontend<br>• Implement visualizations<br>• Conduct experiments | Full-stack app working locally |
| **Week 4** | • Deploy to cloud platform<br>• Write technical report<br>• Create presentation<br>• Final testing | Complete project ready for submission |

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x_t$ | Input at time step $t$ |
| $h_t$ | Hidden state at time $t$ |
| $C_t$ | Cell state at time $t$ (LSTM) |
| $y_t$ | Output at time $t$ |
| $W$ | Weight matrix |
| $b$ | Bias vector |
| $\sigma$ | Sigmoid activation function |
| $\tanh$ | Hyperbolic tangent activation function |
| $\odot$ | Element-wise multiplication (Hadamard product) |
| $T$ | Total number of time steps |
| $d$ | Input dimension |
| $h$ | Hidden state dimension |
| $o$ | Output dimension |
| $\mathcal{L}$ | Loss function |

---

## Getting Help

If you encounter issues:

1. **Check the Troubleshooting Guide** above
2. **Review the documentation** for the specific library (TensorFlow, FastAPI, React)
3. **Search Stack Overflow** for error messages
4. **Ask your instructor** during office hours
5. **Collaborate with classmates** (but write your own code!)

**Important**: Understanding *why* your code works is more important than just getting it to work. Take time to understand the mathematics and architecture.

---

**Good luck building your RNN text generator! This project will give you deep insight into how modern language models work.**