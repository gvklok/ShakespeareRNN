# Shakespeare RNN Text Generator

A full-stack deep learning application that generates Shakespearean-style text using LSTM (Long Short-Term Memory) neural networks trained on the complete works of William Shakespeare.

## 🚀 Live Demo

- **Frontend Application**: https://shakespearernn.onrender.com/
- **API Backend**: https://shakespearernn-3.onrender.com/
- **API Documentation (Swagger UI)**: https://shakespearernn-3.onrender.com/docs

## 📖 Project Overview

This project implements a character-level text generation system using Recurrent Neural Networks (RNNs) with LSTM architecture. The model was trained on Shakespeare's complete works to learn the patterns, vocabulary, and stylistic elements of Early Modern English literature.

### Key Features

- **Text Generation**: Generate Shakespeare-style text from any seed phrase
- **Temperature Control**: Adjust creativity vs coherence (0.5 = conservative, 2.0 = creative)
- **Training Metrics Visualization**: Interactive charts showing loss and accuracy over epochs
- **Model Information Dashboard**: View architecture details, parameters, and vocabulary size
- **REST API**: FastAPI backend with full CORS support
- **Modern Web Interface**: React + TypeScript frontend with real-time generation

## 🎯 Model Performance

**Final Model Statistics:**
- **Training Accuracy**: 15.85%
- **Validation Accuracy**: 11.99%
- **Validation Perplexity**: 345.70
- **Vocabulary Size**: ~3,000 unique words
- **Training Epochs**: 12
- **Total Parameters**: ~1.5M
- **Model Size**: 24MB

### Training Configuration
- **Architecture**: 2-layer LSTM with 256 hidden units
- **Embedding Dimension**: 100
- **Dropout Rate**: 0.2
- **Batch Size**: 64
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Training Data**: Complete Works of Shakespeare (~5.4MB text)

## 🔬 Experiments Conducted

### Experiment 1: Vocabulary Size Impact
Compared models with 3K vs 5K vocabulary sizes:
- **3K Vocab**: Val Perplexity 345.70 (selected model)
- **5K Vocab**: Val Perplexity 1129.71 (overfitting)
- **Conclusion**: Smaller vocabulary generalized better

### Experiment 2: Temperature Analysis
Generated text at various temperatures to analyze creativity vs coherence:
- **T=0.5**: Highly repetitive but grammatically correct
- **T=1.0**: Balanced creativity and coherence
- **T=1.5**: More creative, occasional errors
- **T=2.0**: Highly creative but less coherent

See `TEMPERATURE_EXPERIMENT.md` and `MODEL_COMPARISON.md` for detailed findings.

## 🏗️ Architecture

### Backend (Python + PyTorch)
```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── text_generator.py    # LSTM model class
│   ├── train.py             # Training script
│   └── models.py            # Pydantic schemas
├── saved_models/            # Trained model files
├── visualizations/          # Training plots
└── requirements.txt
```

### Frontend (React + TypeScript)
```
frontend/
├── src/
│   ├── components/
│   │   ├── TextGenerator.tsx       # Text generation interface
│   │   ├── TrainingMetrics.tsx     # Training charts
│   │   └── ModelVisualizer.tsx     # Model info display
│   ├── services/
│   │   └── api.ts                  # API client
│   └── App.tsx                     # Main app component
└── package.json
```

## 🛠️ Technology Stack

### Backend
- **PyTorch 2.0+** - Deep learning framework
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization

### Frontend
- **React 18** - UI library
- **TypeScript** - Type-safe JavaScript
- **Axios** - HTTP client
- **Recharts** - Interactive charts
- **Tailwind CSS** - Utility-first CSS

### Deployment
- **Render** - Cloud platform (backend + frontend)
- **Docker** - Containerization (optional)

## 📋 API Endpoints

### Core Endpoints

**Health Check**
```
GET /
Response: { "status": "ok", "model_status": "loaded", "framework": "PyTorch" }
```

**Generate Text**
```
POST /generate
Body: {
  "seed_text": "to be or not to be",
  "length": 100,
  "temperature": 1.0
}
Response: { "generated_text": "..." }
```

**Training Metrics**
```
GET /metrics
Response: {
  "loss": [5.89, 5.41, ...],
  "accuracy": [0.07, 0.10, ...],
  "val_loss": [...],
  "val_accuracy": [...],
  "epochs": 12
}
```

**Model Information**
```
GET /model-info
Response: {
  "status": "Model loaded",
  "total_params": 1500000,
  "vocab_size": 3000,
  "max_sequence_length": 50
}
```

## 🚀 Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 16+
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/gvklok/ShakespeareRNN.git
cd ShakespeareRNN/backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm start
```

Frontend will open at `http://localhost:3000`

## 🎓 Training Your Own Model

1. **Prepare training data**
   - Add your text corpus to `backend/data/training_text.txt`
   - Minimum 100KB recommended

2. **Configure training parameters** (in `backend/app/train.py`):
```python
config = {
    'embedding_dim': 100,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 12,
    'learning_rate': 0.001
}
```

3. **Train the model**
```bash
cd backend
python -m app.train
```

4. **Monitor training progress**
   - Training metrics saved to `saved_models/training_history.pkl`
   - Visualizations generated in `visualizations/`

## 📊 Model Architecture Details

```
LSTMTextGenerator(
  (embedding): Embedding(3000, 100, padding_idx=0)
  (lstm): LSTM(100, 256, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2)
  (fc): Linear(in_features=256, out_features=3000)
)
```

**Total Parameters**: ~1.5 million
- Embedding: 300K params
- LSTM layers: ~1M params
- Fully connected: ~770K params

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individually
docker build -t shakespeare-backend ./backend
docker build -t shakespeare-frontend ./frontend
```

## 📝 Project Deliverables

- ✅ Trained LSTM model with documented performance
- ✅ Full-stack web application (deployed)
- ✅ REST API with comprehensive documentation
- ✅ Training visualizations and metrics
- ✅ Experimental analysis (2 experiments)
- ✅ Generated text samples
- ✅ Technical documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**gvklok**
- GitHub: [@gvklok](https://github.com/gvklok)
- Project: [ShakespeareRNN](https://github.com/gvklok/ShakespeareRNN)

## 🙏 Acknowledgments

- Training data: Complete Works of William Shakespeare from Project Gutenberg
- Framework: PyTorch deep learning framework
- Assignment: CST-435 Recurrent Neural Networks Activity

---

**Note**: This project was developed as part of a machine learning course assignment focusing on RNN text generation and full-stack ML application development.
