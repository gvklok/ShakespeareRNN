/**
 * Training Metrics Component
 */

import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { getTrainingMetrics, TrainingMetrics as MetricsType } from '../services/api';

const TrainingMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsType | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const data = await getTrainingMetrics();
      setMetrics(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <p>Loading metrics...</p>
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

  if (!metrics || metrics.epochs === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-gray-500">No training metrics available. Train the model first.</p>
      </div>
    );
  }

  // Prepare data for charts
  const lossData = metrics.loss.map((loss, index) => ({
    epoch: index + 1,
    loss,
    val_loss: metrics.val_loss?.[index],
  }));

  const accuracyData = metrics.accuracy?.map((acc, index) => ({
    epoch: index + 1,
    accuracy: acc,
    val_accuracy: metrics.val_accuracy?.[index],
  })) || [];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Training Metrics</h2>

      <div className="mb-4">
        <p className="text-sm text-gray-600">Total Epochs: {metrics.epochs}</p>
      </div>

      {/* Loss Chart */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-2">Loss</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lossData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
            {metrics.val_loss && (
              <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Validation Loss" />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy Chart */}
      {accuracyData.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-2">Accuracy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Training Accuracy" />
              {metrics.val_accuracy && (
                <Line type="monotone" dataKey="val_accuracy" stroke="#82ca9d" name="Validation Accuracy" />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <button
        onClick={fetchMetrics}
        className="mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
      >
        Refresh Metrics
      </button>
    </div>
  );
};

export default TrainingMetrics;
