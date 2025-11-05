"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequest(BaseModel):
    """Request model for text generation"""
    seed_text: str = Field(..., min_length=1, description="Starting text for generation")
    length: int = Field(default=100, ge=1, le=1000, description="Number of characters to generate")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")

class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str

class TrainingMetrics(BaseModel):
    """Model for training metrics"""
    loss: List[float]
    accuracy: Optional[List[float]] = None
    val_loss: Optional[List[float]] = None
    val_accuracy: Optional[List[float]] = None
    epochs: int
