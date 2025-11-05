/**
 * Model Visualizer Component
 */

import React, { useEffect, useState } from 'react';
import { getModelInfo, ModelInfo } from '../services/api';

const ModelVisualizer: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const data = await getModelInfo();
      setModelInfo(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch model info');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <p>Loading model information...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Model Information</h2>

      <div className="space-y-3">
        <div className="flex justify-between items-center py-2 border-b">
          <span className="font-medium text-gray-700">Status:</span>
          <span className={`px-3 py-1 rounded-full text-sm ${
            modelInfo?.status === 'Model loaded'
              ? 'bg-green-100 text-green-800'
              : 'bg-yellow-100 text-yellow-800'
          }`}>
            {modelInfo?.status}
          </span>
        </div>

        {modelInfo?.total_params !== undefined && (
          <div className="flex justify-between items-center py-2 border-b">
            <span className="font-medium text-gray-700">Total Parameters:</span>
            <span className="text-gray-900">
              {modelInfo.total_params.toLocaleString()}
            </span>
          </div>
        )}

        {modelInfo?.vocab_size !== undefined && (
          <div className="flex justify-between items-center py-2 border-b">
            <span className="font-medium text-gray-700">Vocabulary Size:</span>
            <span className="text-gray-900">
              {modelInfo.vocab_size.toLocaleString()}
            </span>
          </div>
        )}

        {modelInfo?.max_sequence_length !== undefined && (
          <div className="flex justify-between items-center py-2 border-b">
            <span className="font-medium text-gray-700">Max Sequence Length:</span>
            <span className="text-gray-900">
              {modelInfo.max_sequence_length}
            </span>
          </div>
        )}
      </div>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Model Architecture</h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="space-y-2 text-sm">
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 1:</div>
              <div className="text-gray-600">Embedding Layer</div>
            </div>
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 2:</div>
              <div className="text-gray-600">LSTM (with return sequences)</div>
            </div>
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 3:</div>
              <div className="text-gray-600">Dropout (0.2)</div>
            </div>
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 4:</div>
              <div className="text-gray-600">LSTM</div>
            </div>
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 5:</div>
              <div className="text-gray-600">Dropout (0.2)</div>
            </div>
            <div className="flex items-center">
              <div className="w-32 font-medium text-gray-700">Layer 6:</div>
              <div className="text-gray-600">Dense (softmax activation)</div>
            </div>
          </div>
        </div>
      </div>

      <button
        onClick={fetchModelInfo}
        className="mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
      >
        Refresh Info
      </button>
    </div>
  );
};

export default ModelVisualizer;
