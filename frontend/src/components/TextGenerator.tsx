/**
 * Text Generator Component
 */

import React, { useState } from 'react';
import { generateText, GenerateTextRequest } from '../services/api';

const TextGenerator: React.FC = () => {
  const [seedText, setSeedText] = useState<string>('');
  const [length, setLength] = useState<number>(100);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [generatedText, setGeneratedText] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter seed text');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const request: GenerateTextRequest = {
        seed_text: seedText,
        length,
        temperature,
      };

      const response = await generateText(request);
      setGeneratedText(response.generated_text);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Text Generator</h2>

      <div className="space-y-4">
        {/* Seed Text Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Seed Text
          </label>
          <input
            type="text"
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter starting text..."
          />
        </div>

        {/* Length Slider */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Length: {length} characters
          </label>
          <input
            type="range"
            min="10"
            max="500"
            value={length}
            onChange={(e) => setLength(Number(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Temperature Slider */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Temperature: {temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Lower = more predictable, Higher = more random
          </p>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? 'Generating...' : 'Generate Text'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {/* Generated Text Output */}
        {generatedText && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Generated Text
            </label>
            <div className="bg-gray-50 border border-gray-300 rounded-md p-4 min-h-[150px]">
              <p className="whitespace-pre-wrap">{generatedText}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextGenerator;
