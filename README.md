# RNN Text Generator

A full-stack application for training and generating text using Recurrent Neural Networks (RNN) with LSTM layers.

**Framework**: This project uses **PyTorch** for deep learning. The model architecture, training loop, and inference are all implemented using PyTorch's flexible and intuitive API.

## Project Structure

```
rnn-text-generator/
│
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application
│   │   ├── models.py          # Pydantic models
│   │   ├── text_generator.py # RNN model class
│   │   └── train.py           # Training script
│   ├── data/
│   │   └── training_text.txt  # Training corpus
│   ├── saved_models/          # Trained models
│   ├── visualizations/        # Training plots
│   └── requirements.txt
│
├── frontend/                   # React TypeScript frontend
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

## Features

- **Text Generation**: Generate text based on seed input using trained RNN model
- **Training Metrics Visualization**: View loss and accuracy charts from training
- **Model Information**: Display model architecture and parameters
- **Temperature Control**: Adjust randomness of text generation
- **REST API**: FastAPI backend with CORS support
- **Modern UI**: React with TypeScript and Tailwind CSS

## Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your training text to `data/training_text.txt`

5. Train the model:
```bash
python -m app.train
```

6. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Using Docker

Alternatively, you can run the entire application using Docker Compose:

```bash
docker-compose up --build
```

This will start both the backend and frontend services.

## API Endpoints

- `GET /` - Health check
- `POST /generate` - Generate text
  - Body: `{ "seed_text": string, "length": number, "temperature": number }`
- `GET /metrics` - Get training metrics
- `GET /model-info` - Get model information

## Training Your Own Model

1. Prepare your training text corpus and save it to `backend/data/training_text.txt`

2. Modify training parameters in `backend/app/train.py` if needed:
   - `embedding_dim`: Size of embedding layer (default: 100)
   - `lstm_units`: Number of LSTM units (default: 150)
   - `epochs`: Number of training epochs (default: 100)
   - `batch_size`: Training batch size (default: 128)

3. Run the training script:
```bash
cd backend
python -m app.train
```

4. Monitor training progress in TensorBoard:
```bash
tensorboard --logdir=visualizations/logs
```

## Technologies Used

### Backend
- **PyTorch 2.1.0** (Deep Learning Framework)
- FastAPI 0.109.0
- Uvicorn
- NumPy
- Matplotlib/Seaborn for visualization
- TensorBoard for training visualization
- tqdm for progress bars

### Frontend
- React 18
- TypeScript
- Axios for API calls
- Recharts for data visualization
- Tailwind CSS for styling

## Model Architecture

The PyTorch LSTM model consists of:
1. **Embedding Layer**: Converts word indices to dense vectors
2. **LSTM Layers**: 2-layer stacked LSTM with configurable hidden dimensions
3. **Dropout**: Applied between LSTM layers for regularization
4. **Fully Connected Layer**: Maps LSTM output to vocabulary size
5. **Softmax**: Applied during prediction for probability distribution

Key features:
- Automatic GPU detection and usage when available
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Model checkpointing to save best performing model

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!
