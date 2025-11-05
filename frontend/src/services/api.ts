/**
 * API service for communicating with the backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GenerateTextRequest {
  seed_text: string;
  length: number;
  temperature: number;
}

export interface GenerateTextResponse {
  generated_text: string;
}

export interface TrainingMetrics {
  loss: number[];
  accuracy?: number[];
  val_loss?: number[];
  val_accuracy?: number[];
  epochs: number;
}

export interface ModelInfo {
  status: string;
  total_params?: number;
  vocab_size?: number;
  max_sequence_length?: number;
}

/**
 * Generate text using the trained model
 */
export const generateText = async (
  request: GenerateTextRequest
): Promise<GenerateTextResponse> => {
  const response = await api.post<GenerateTextResponse>('/generate', request);
  return response.data;
};

/**
 * Get training metrics
 */
export const getTrainingMetrics = async (): Promise<TrainingMetrics> => {
  const response = await api.get<TrainingMetrics>('/metrics');
  return response.data;
};

/**
 * Get model information
 */
export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get<ModelInfo>('/model-info');
  return response.data;
};

/**
 * Health check
 */
export const healthCheck = async (): Promise<{ status: string; message: string }> => {
  const response = await api.get('/');
  return response.data;
};

export default api;
