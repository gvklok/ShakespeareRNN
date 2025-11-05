"""
FastAPI Application for RNN Text Generator (PyTorch)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .models import GenerateRequest, GenerateResponse, TrainingMetrics
from .text_generator import RNNTextGenerator
import os

app = FastAPI(
    title="RNN Text Generator API (PyTorch)",
    description="API for training and generating text using RNN models with PyTorch",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # React dev server + allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize text generator
generator = RNNTextGenerator(
    model_path="saved_models/model.pt",
    tokenizer_path="saved_models/tokenizer.pkl"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "loaded" if generator.model is not None else "not loaded"
    return {
        "status": "ok",
        "message": "RNN Text Generator API (PyTorch) is running",
        "model_status": model_status,
        "framework": "PyTorch"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the trained RNN model"""
    try:
        generated_text = generator.generate_text(
            seed_text=request.seed_text,
            length=request.length,
            temperature=request.temperature
        )
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=TrainingMetrics)
async def get_metrics():
    """Get training metrics"""
    try:
        metrics = generator.get_training_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    try:
        info = generator.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/training")
async def get_training_visualization():
    """Get training history visualization"""
    viz_path = "visualizations/training_history.png"
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Training visualization not found")
    return FileResponse(viz_path)
