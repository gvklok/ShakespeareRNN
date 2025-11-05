# Setup Instructions for PyTorch RNN Text Generator

Follow these steps to set up your environment and train your first model.

## Step 1: Create Virtual Environment

```bash
cd /Users/gvklok/Documents/CST-435/RNNs/backend
python3 -m venv venv
```

## Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

## Step 3: Install PyTorch and Dependencies

For **Apple Silicon (M1/M2/M3)** Macs:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install PyTorch with MPS (Metal Performance Shaders) support, which allows you to use your Apple Silicon GPU for accelerated training.

## Step 4: Verify Installation

```bash
python test_setup.py
```

This will check:
- ✓ PyTorch installation
- ✓ MPS (Apple Silicon GPU) availability
- ✓ Training data exists
- ✓ Required directories exist

Expected output:
```
============================================================
PyTorch Setup Test
============================================================

✓ PyTorch version: 2.1.0

CUDA available: False

MPS (Apple Silicon) available: True
  ✓ MPS is available - will use Apple Silicon GPU for acceleration

============================================================
Training will use: Apple Silicon GPU (MPS)
Device: mps
============================================================
```

## Step 5: Train Your Model

```bash
python -m app.train
```

This will:
- Load the Romeo and Juliet text from `data/training_text.txt`
- Preprocess and tokenize the text
- Build the LSTM model
- Train for up to 100 epochs (with early stopping)
- Save the best model to `saved_models/model.pt`
- Generate training visualizations in `visualizations/`

Training will show progress bars like this:
```
Using Apple Silicon GPU (MPS)
Loading training data...
Total vocabulary size: 3487

Building model...
============================================================
LSTMTextGenerator(...)
============================================================
Total parameters: 1,234,567
Trainable parameters: 1,234,567
============================================================

Starting training...
============================================================
Epoch 1/100 [Train]: 100%|████████| 45/45 [00:15<00:00, loss=5.2341, acc=15.23%]
Epoch 1/100 [Val]  : 100%|████████| 5/5 [00:01<00:00, loss=4.9876, acc=18.45%]

Epoch 1/100:
  Train Loss: 5.2341, Train Acc: 15.23%
  Val Loss: 4.9876, Val Acc: 18.45%
  ✓ Model saved (best val_loss: 4.9876)
============================================================
```

**Training Tips:**
- Training on Apple Silicon GPU (MPS) should take 5-15 minutes for Romeo and Juliet
- The model will automatically stop early if it's not improving
- Watch for the validation loss to decrease - that's a sign of good learning

## Step 6: Start the API Server

Once training is complete:

```bash
uvicorn app.main:app --reload
```

The API will be available at: http://localhost:8000

Test it by visiting: http://localhost:8000 in your browser. You should see:
```json
{
  "status": "ok",
  "message": "RNN Text Generator API (PyTorch) is running",
  "model_status": "loaded",
  "framework": "PyTorch"
}
```

## Step 7: Test Text Generation

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_text": "romeo wherefore art thou",
    "length": 50,
    "temperature": 1.0
  }'
```

Or visit the API docs at: http://localhost:8000/docs

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
**Solution**: Make sure you activated the virtual environment:
```bash
source venv/bin/activate
```

### "MPS backend out of memory"
**Solution**: Reduce batch size in `app/train.py`:
```python
trainer.train(epochs=100, batch_size=64)  # Instead of 128
```

### Training is very slow
**Solution**:
1. Check if MPS is being used: Look for "Using Apple Silicon GPU (MPS)" message
2. Make sure you're not running other heavy applications
3. Reduce model size if needed

### "RuntimeError: MPS backend is not available"
**Solution**: Make sure you're on macOS 12.3+ with Apple Silicon. If on Intel Mac or older macOS, the model will automatically fall back to CPU.

## Next Steps

1. **Experiment with generation**:
   - Try different seed texts
   - Adjust temperature (0.5 = more predictable, 2.0 = more creative)
   - Change generation length

2. **Improve the model**:
   - Train on more/different data
   - Adjust hyperparameters (LSTM units, layers, dropout)
   - Train for more epochs

3. **Set up the frontend** (optional):
   - See the main README.md for frontend setup instructions

## Need Help?

- Check `test_setup.py` output for diagnostic info
- Review error messages carefully
- Consult the PYTORCH_MIGRATION.md guide
- Ask your instructor

---

**Happy training! 🚀**
