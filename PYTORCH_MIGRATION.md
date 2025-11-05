# PyTorch Migration Guide

This document explains the changes made to migrate the RNN Text Generator from TensorFlow to PyTorch.

## Summary of Changes

The project has been fully restructured to use PyTorch instead of TensorFlow/Keras. All core functionality remains the same, but the implementation is now using PyTorch's native APIs.

## Key Differences

### 1. Model Architecture

**Before (TensorFlow/Keras):**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(lstm_units, return_sequences=True),
    Dropout(0.2),
    LSTM(lstm_units),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])
```

**After (PyTorch):**
```python
import torch.nn as nn

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

### 2. Training Loop

**TensorFlow** uses a high-level `model.fit()` API:
```python
model.fit(X, y, epochs=100, batch_size=128)
```

**PyTorch** requires explicit training loops:
```python
for epoch in range(epochs):
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 3. Model Saving/Loading

**Before (TensorFlow):**
```python
model.save('model.h5')
model = load_model('model.h5')
```

**After (PyTorch):**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model.pt')

checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 4. Data Handling

**PyTorch** uses custom Dataset and DataLoader classes:
```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
```

### 5. GPU Support

**PyTorch** provides explicit device management:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)
```

## File Changes

### Updated Files

1. **`backend/requirements.txt`**
   - Replaced `tensorflow==2.15.0` with `torch==2.1.0` and `torchvision==0.16.0`
   - Added `tqdm==4.66.1` for progress bars

2. **`backend/app/text_generator.py`**
   - Complete rewrite using PyTorch `nn.Module`
   - Custom `LSTMTextGenerator` class
   - Custom `Tokenizer` class (since PyTorch doesn't have built-in text preprocessing)
   - Device-aware inference

3. **`backend/app/train.py`**
   - Explicit training loop with progress bars
   - PyTorch DataLoader for batch processing
   - Learning rate scheduling with `ReduceLROnPlateau`
   - Gradient clipping for stability
   - Early stopping implementation
   - TensorBoard integration for PyTorch

4. **`backend/app/main.py`**
   - Updated to load PyTorch models (`.pt` instead of `.h5`)
   - Model info endpoint now shows PyTorch-specific details

5. **`RNN_Activity_Guide.md`**
   - Updated technology stack section
   - Updated learning objectives
   - Updated recommended resources
   - Updated tools and libraries section

6. **`README.md`**
   - Updated technologies section
   - Added PyTorch-specific features
   - Updated model architecture description

## Installation Instructions

### 1. Set up a fresh virtual environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have a CUDA-capable GPU, you may want to install the GPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Prepare training data

Place your training text in `backend/data/training_text.txt`

### 4. Train the model

```bash
cd backend
python -m app.train
```

This will:
- Load and preprocess your text data
- Build the LSTM model
- Train with progress bars showing loss and accuracy
- Save the best model to `saved_models/model.pt`
- Generate training visualizations in `visualizations/`

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Benefits of PyTorch

### 1. **More Flexible and Pythonic**
   - Native Python control flow
   - Easy to debug with standard Python debuggers
   - More intuitive for complex architectures

### 2. **Better for Research and Experimentation**
   - Dynamic computational graphs
   - Easier to modify models on-the-fly
   - Widely used in research community

### 3. **Explicit is Better than Implicit**
   - Full control over training loop
   - Clear understanding of what's happening
   - Easier to customize and extend

### 4. **Strong Community Support**
   - Extensive tutorials and documentation
   - Active development
   - Wide adoption in academia and industry

### 5. **GPU Support**
   - Seamless CPU/GPU switching
   - Automatic memory management
   - Mixed precision training support

## Testing Your Setup

### 1. Quick Test

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2. Test the Text Generator

```python
from app.text_generator import RNNTextGenerator

generator = RNNTextGenerator()
if generator.model is not None:
    text = generator.generate_text("Once upon a time", length=50)
    print(text)
else:
    print("No model loaded. Train a model first!")
```

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Make sure you've activated the virtual environment and installed requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in training:
```python
trainer.train(batch_size=64)  # Instead of 128
```

### Issue: Training is slow
**Solutions**:
1. Check if GPU is being used: `print(torch.cuda.is_available())`
2. Install CUDA-enabled PyTorch
3. Reduce model size (fewer LSTM units or layers)
4. Reduce sequence length

### Issue: Model generates poor quality text
**Solutions**:
1. Train for more epochs
2. Use a larger training dataset
3. Increase model capacity (more LSTM units)
4. Experiment with different temperatures during generation

## Next Steps

1. **Experiment with hyperparameters**: Try different embedding dimensions, LSTM units, and number of layers
2. **Try different datasets**: Download books from Project Gutenberg or use different text sources
3. **Visualize training**: Use TensorBoard to monitor training progress
4. **Deploy your model**: Consider deploying to Render, Railway, or Hugging Face Spaces

## Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch for Deep Learning](https://www.learnpytorch.io/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Questions?

If you encounter any issues or have questions about the PyTorch implementation, please:
1. Check the PyTorch documentation
2. Review the code comments in the source files
3. Consult the RNN Activity Guide for detailed explanations
4. Ask your instructor for help

---

**Happy training with PyTorch! 🔥**
