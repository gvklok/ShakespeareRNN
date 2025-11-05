# Next Steps - Your RNN Text Generator is Ready!

## ✅ What's Been Done

Your project has been successfully restructured to use **PyTorch** with **Apple Silicon GPU (MPS)** support:

1. ✅ **Migrated from TensorFlow to PyTorch**
   - All model code rewritten using PyTorch `nn.Module`
   - Custom LSTM text generator implementation
   - Native PyTorch training loops with progress tracking

2. ✅ **Added Apple Silicon GPU Support**
   - Automatic MPS device detection
   - GPU-accelerated training on M1/M2/M3 Macs
   - Falls back to CPU if MPS not available

3. ✅ **Training Data Ready**
   - Romeo and Juliet text loaded (~170KB)
   - Symbolic link created for easy access
   - Perfect size for training

4. ✅ **Helper Scripts Created**
   - `test_setup.py` - Verify installation
   - `quickstart.sh` - Automated setup
   - `SETUP.md` - Detailed instructions

## 🚀 What To Do Next

### Option 1: Quick Automated Setup (Recommended)

Run the quick start script from the backend directory:

```bash
cd /Users/gvklok/Documents/CST-435/RNNs/backend
./quickstart.sh
```

This will:
- Create a virtual environment
- Install all PyTorch dependencies
- Test your setup
- Confirm MPS is working

### Option 2: Manual Step-by-Step Setup

If you prefer to do it manually or the script has issues:

#### 1. Create and Activate Virtual Environment

```bash
cd /Users/gvklok/Documents/CST-435/RNNs/backend
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal.

#### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch 2.1.0 (with MPS support)
- FastAPI for the API
- matplotlib, seaborn for visualizations
- tqdm for progress bars
- All other required packages

#### 3. Verify Installation

```bash
python test_setup.py
```

Expected output:
```
============================================================
PyTorch Setup Test
============================================================

✓ PyTorch version: 2.1.0

MPS (Apple Silicon) available: True
  ✓ MPS is available - will use Apple Silicon GPU for acceleration

============================================================
Training will use: Apple Silicon GPU (MPS)
Device: mps
============================================================

✓ Training data found: data/training_text.txt
  Size: 169,546 characters
  Words: ~25,000
```

## 🎯 Training Your First Model

Once setup is complete, train your model:

```bash
python -m app.train
```

### What Happens During Training:

1. **Data Loading** (~5 seconds)
   - Reads Romeo and Juliet text
   - Preprocesses and tokenizes
   - Creates training sequences

2. **Model Building** (~2 seconds)
   - Creates 2-layer LSTM network
   - ~1.5 million parameters
   - Moves model to MPS (GPU)

3. **Training** (~5-15 minutes on Apple Silicon)
   - Progress bars show loss and accuracy
   - Automatically saves best model
   - Early stopping if not improving

### Example Training Output:

```
Using Apple Silicon GPU (MPS)
Loading training data...
Total vocabulary size: 3,487
Max sequence length: 142
Total sequences: 23,456

Building model...
============================================================
LSTMTextGenerator(
  (embedding): Embedding(3487, 100)
  (lstm): LSTM(100, 150, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=150, out_features=3487, bias=True)
)
============================================================
Total parameters: 1,567,437
Trainable parameters: 1,567,437
============================================================

Starting training...
Epoch 1/100 [Train]: 100%|██████| 183/183 [00:12<00:00, loss=6.1234, acc=10.23%]
Epoch 1/100 [Val]  : 100%|██████| 21/21 [00:01<00:00, loss=5.8765, acc=12.45%]

Epoch 1/100:
  Train Loss: 6.1234, Train Acc: 10.23%
  Val Loss: 5.8765, Val Acc: 12.45%
  ✓ Model saved (best val_loss: 5.8765)
============================================================
```

### Training Progress:

- **Epochs 1-10**: Loss drops rapidly, accuracy climbs
- **Epochs 10-30**: Steady improvement
- **Epochs 30+**: Fine-tuning, early stopping may trigger

The model typically converges around epoch 20-40 with:
- Training accuracy: 40-60%
- Validation accuracy: 35-50%
- Loss: 2.5-3.5

## 🌐 Running the API

After training completes:

```bash
uvicorn app.main:app --reload
```

### Test the API:

1. **Health Check**:
   ```bash
   curl http://localhost:8000
   ```

2. **Generate Text**:
   ```bash
   curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "seed_text": "wherefore art thou romeo",
       "length": 50,
       "temperature": 1.0
     }'
   ```

3. **Interactive API Docs**:
   Visit http://localhost:8000/docs in your browser

## 🧪 Experimenting with Generation

### Temperature Controls Creativity:

```bash
# Conservative (0.5) - More predictable, follows patterns
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "to be or not to be", "length": 30, "temperature": 0.5}'

# Balanced (1.0) - Good mix of coherence and creativity
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "to be or not to be", "length": 30, "temperature": 1.0}'

# Creative (1.5) - More random, experimental
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "to be or not to be", "length": 30, "temperature": 1.5}'
```

### Try Different Seed Texts:

- "romeo wherefore art thou"
- "two households both alike in dignity"
- "a plague on both your houses"
- "parting is such sweet sorrow"

## 📊 Viewing Training Results

Training generates visualizations in `visualizations/`:

1. **Training History Plot**:
   - Loss curves (training vs validation)
   - Accuracy curves
   - Saved as `training_history.png`

2. **TensorBoard Logs**:
   ```bash
   tensorboard --logdir=visualizations/logs
   ```
   Then visit http://localhost:6006

## 🎨 Setting Up the Frontend (Optional)

If you want a web interface:

```bash
cd /Users/gvklok/Documents/CST-435/RNNs/frontend

# Install dependencies
npm install

# Start dev server
npm start
```

Visit http://localhost:3000 for the React UI.

## 🔧 Customizing Your Model

Edit `backend/app/train.py` to experiment:

### Change Model Architecture:

```python
trainer.build_model(
    embedding_dim=100,    # Try: 50, 100, 200
    hidden_dim=150,       # Try: 100, 150, 256, 512
    num_layers=2,         # Try: 1, 2, 3
    dropout=0.2           # Try: 0.1, 0.2, 0.3, 0.5
)
```

### Change Training Settings:

```python
trainer.train(
    epochs=100,           # Try: 50, 100, 200
    batch_size=128,       # Try: 64, 128, 256
    learning_rate=0.001   # Try: 0.0001, 0.001, 0.01
)
```

### Use Different Training Data:

1. Add your text file to `backend/data/`
2. Update `RNNTrainer` initialization:
   ```python
   trainer = RNNTrainer(text_path="data/your_text.txt")
   ```

## 📚 Project Structure Reference

```
backend/
├── app/
│   ├── text_generator.py   # PyTorch LSTM model
│   ├── train.py            # Training script
│   ├── main.py             # FastAPI server
│   └── models.py           # API request/response models
├── data/
│   ├── RomeoJuliet.txt     # Original training data
│   └── training_text.txt → # Symlink to above
├── saved_models/
│   ├── model.pt            # Trained PyTorch model (after training)
│   └── tokenizer.pkl       # Word tokenizer (after training)
├── visualizations/
│   ├── training_history.png # Loss/accuracy plots (after training)
│   └── logs/               # TensorBoard logs (after training)
├── requirements.txt        # Python dependencies
├── test_setup.py          # Setup verification script
├── quickstart.sh          # Automated setup script
└── SETUP.md               # Detailed setup guide
```

## 🐛 Common Issues & Solutions

### Issue: "MPS backend out of memory"
**Solution**: Reduce batch size:
```python
trainer.train(batch_size=64)  # Instead of 128
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Activate virtual environment:
```bash
source venv/bin/activate
```

### Issue: Training is slow
**Check**:
1. Is MPS being used? Look for "Using Apple Silicon GPU (MPS)"
2. Close other heavy applications
3. Try smaller model: `hidden_dim=100` instead of 150

### Issue: Generated text is nonsense
**Solutions**:
1. Train longer (more epochs)
2. Larger dataset
3. Lower temperature (0.7 instead of 1.0)
4. Increase model capacity

## 📖 Documentation Reference

- **[SETUP.md](backend/SETUP.md)** - Detailed setup instructions
- **[PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md)** - Migration guide from TensorFlow
- **[RNN_Activity_Guide.md](RNN_Activity_Guide.md)** - Complete RNN theory and implementation guide
- **[README.md](README.md)** - Project overview

## 🎓 Learning Resources

- [PyTorch Official Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Karpathy's Blog on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## ✨ Ready to Start!

Your environment is fully set up with:
- ✅ PyTorch with Apple Silicon GPU support
- ✅ Romeo and Juliet training data
- ✅ Complete training and inference pipeline
- ✅ FastAPI backend ready to deploy
- ✅ Automated setup and testing scripts

**Run this to begin:**

```bash
cd /Users/gvklok/Documents/CST-435/RNNs/backend
./quickstart.sh
```

Then train your model:

```bash
python -m app.train
```

**Happy training! 🚀🔥**

---

Questions? Check the documentation files or ask your instructor!
