/**
 * Main App Component
 */

import React, { useEffect, useState } from 'react';
import TextGenerator from './components/TextGenerator';
import ModelVisualizer from './components/ModelVisualizer';
import TrainingMetrics from './components/TrainingMetrics';
import { healthCheck } from './services/api';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'generator' | 'metrics' | 'model'>('generator');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      await healthCheck();
      setBackendStatus('online');
    } catch (err) {
      setBackendStatus('offline');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">RNN Text Generator</h1>
          <p className="text-blue-100 mt-2">Generate text using Recurrent Neural Networks</p>

          {/* Backend Status */}
          <div className="mt-3 flex items-center">
            <span className="text-sm mr-2">Backend Status:</span>
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              backendStatus === 'online'
                ? 'bg-green-500 text-white'
                : backendStatus === 'offline'
                ? 'bg-red-500 text-white'
                : 'bg-yellow-500 text-white'
            }`}>
              {backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking...'}
            </span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white shadow">
        <div className="container mx-auto px-4">
          <nav className="flex space-x-4">
            <button
              onClick={() => setActiveTab('generator')}
              className={`py-4 px-6 font-medium border-b-2 transition-colors ${
                activeTab === 'generator'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              Text Generator
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`py-4 px-6 font-medium border-b-2 transition-colors ${
                activeTab === 'metrics'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              Training Metrics
            </button>
            <button
              onClick={() => setActiveTab('model')}
              className={`py-4 px-6 font-medium border-b-2 transition-colors ${
                activeTab === 'model'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              Model Info
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {backendStatus === 'offline' && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <p className="font-bold">Backend Offline</p>
            <p>Unable to connect to the backend server. Please make sure it's running on http://localhost:8000</p>
            <button
              onClick={checkBackendHealth}
              className="mt-2 bg-red-600 text-white py-1 px-3 rounded text-sm hover:bg-red-700"
            >
              Retry Connection
            </button>
          </div>
        )}

        <div className="max-w-4xl mx-auto">
          {activeTab === 'generator' && <TextGenerator />}
          {activeTab === 'metrics' && <TrainingMetrics />}
          {activeTab === 'model' && <ModelVisualizer />}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm">
            RNN Text Generator - Built with TensorFlow, FastAPI, and React
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
