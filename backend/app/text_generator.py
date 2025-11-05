"""
RNN Text Generator Model Class using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from typing import Optional, Dict, List, Tuple
from collections import Counter
import re

class LSTMTextGenerator(nn.Module):
    """LSTM-based text generator model"""

    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 hidden_dim: int = 150, num_layers: int = 2, dropout: float = 0.2,
                 padding_idx: int = 0):
        super(LSTMTextGenerator, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """Forward pass through the network"""
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        if hidden is None:
            lstm_out, hidden = self.lstm(embeds)
        else:
            lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = self.dropout(lstm_out)

        # Take output from last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Pass through fully connected layer
        out = self.fc(last_output)  # (batch_size, vocab_size)

        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class Tokenizer:
    """Simple word tokenizer for text processing"""

    def __init__(self, max_vocab_size: Optional[int] = None, min_freq: int = 1):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.max_vocab_size = max_vocab_size
        self.min_freq = max(min_freq, 1)

    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from texts

        Args:
            texts: List of text strings to build vocabulary from
        """
        for text in texts:
            words = text.split()
            self.word_counts.update(words)

        # Filter by minimum frequency first
        filtered = [(word, count) for word, count in self.word_counts.items()
                    if count >= self.min_freq]

        if not filtered:
            raise ValueError("Vocabulary is empty after applying frequency threshold.")

        filtered.sort(key=lambda item: item[1], reverse=True)

        # Limit vocabulary to most common words if max_vocab_size is set
        if self.max_vocab_size:
            original_vocab = len(filtered)
            filtered = filtered[:self.max_vocab_size - 1]  # -1 for padding/UNK token
            print(f"  Limiting vocabulary to {self.max_vocab_size} most common words")
            print(f"  (filtered from {original_vocab} tokens after min_freq={self.min_freq})")
        else:
            print(f"  Using {len(filtered)} tokens with min_freq={self.min_freq}")

        # Create word to index mapping (0 is reserved for padding/unknown)
        self.word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(filtered)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            words = text.split()
            sequence = [self.word_to_idx.get(word, 0) for word in words]
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text"""
        words = [self.idx_to_word.get(idx, '') for idx in sequence if idx != 0]
        return ' '.join(words)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including padding token)"""
        return len(self.word_to_idx) + 1

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'min_freq' not in self.__dict__:
            self.min_freq = 1
        if 'max_vocab_size' not in self.__dict__:
            self.max_vocab_size = None


class RNNTextGenerator:
    """RNN-based text generator using PyTorch LSTM"""

    def __init__(self, model_path: str = "saved_models/model.pt",
                 tokenizer_path: str = "saved_models/tokenizer.pkl",
                 device: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Set device with MPS support for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[LSTMTextGenerator] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.max_sequence_len: int = 0
        self.training_history: Optional[Dict] = None
        self.embedding_dim: int = 100
        self.hidden_dim: int = 150
        self.num_layers: int = 2
        self.dropout: float = 0.2

        # Load model if it exists
        self._load_model()

    def _load_model(self):
        """Load trained model and tokenizer"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
                # Load tokenizer
                with open(self.tokenizer_path, 'rb') as f:
                    data = pickle.load(f)
                    self.tokenizer = data['tokenizer']
                    self.max_sequence_len = data['max_sequence_len']
                    self.embedding_dim = data.get('embedding_dim', 100)
                    self.hidden_dim = data.get('hidden_dim', 150)
                    self.num_layers = data.get('num_layers', 2)
                    self.dropout = data.get('dropout', 0.2)

                # Load model
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Initialize model architecture
                self.model = LSTMTextGenerator(
                    vocab_size=self.tokenizer.vocab_size,
                    embedding_dim=self.embedding_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    padding_idx=0
                )

                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                print(f"Model loaded successfully on {self.device}")
            else:
                print("No saved model found")
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^a-z0-9\s.,!?;:'\"\-]", '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def generate_text(self, seed_text: str, length: int = 100,
                     temperature: float = 1.0) -> str:
        """Generate text using the trained model

        Args:
            seed_text: Starting text
            length: Number of words to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please train the model first.")

        self.model.eval()

        # Preprocess seed text
        seed_text = self.preprocess_text(seed_text)
        generated_text = seed_text

        with torch.no_grad():
            for _ in range(length):
                # Tokenize current text
                sequence = self.tokenizer.texts_to_sequences([generated_text])[0]

                # Take last max_sequence_len-1 tokens
                if len(sequence) > self.max_sequence_len - 1:
                    sequence = sequence[-(self.max_sequence_len - 1):]

                # Convert to tensor
                input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

                # Get prediction
                output, _ = self.model(input_tensor)

                # Apply temperature
                output = output / temperature
                probabilities = torch.softmax(output, dim=-1)

                # Sample from the distribution
                probabilities = probabilities.cpu().numpy()[0]
                predicted_idx = np.random.choice(len(probabilities), p=probabilities)

                # Get the word
                predicted_word = self.tokenizer.idx_to_word.get(predicted_idx, '')

                if predicted_word:
                    generated_text += " " + predicted_word

        return generated_text

    def get_training_metrics(self) -> Dict:
        """Get training metrics"""
        if self.training_history is None:
            # Try to load from file
            metrics_path = "saved_models/training_history.pkl"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'rb') as f:
                    self.training_history = pickle.load(f)

        if self.training_history:
            return {
                "loss": self.training_history.get('loss', []),
                "accuracy": self.training_history.get('accuracy', []),
                "val_loss": self.training_history.get('val_loss', []),
                "val_accuracy": self.training_history.get('val_accuracy', []),
                "perplexity": self.training_history.get('perplexity', []),
                "val_perplexity": self.training_history.get('val_perplexity', []),
                "grad_norm": self.training_history.get('grad_norm', []),
                "learning_rate": self.training_history.get('learning_rate', []),
                "epochs": len(self.training_history.get('loss', []))
            }
        else:
            return {
                "loss": [],
                "accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "perplexity": [],
                "val_perplexity": [],
                "grad_norm": [],
                "learning_rate": [],
                "epochs": 0
            }

    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "No model loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "Model loaded",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "max_sequence_length": self.max_sequence_len,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_vocab_size": getattr(self.tokenizer, 'max_vocab_size', None),
            "min_word_freq": getattr(self.tokenizer, 'min_freq', None),
            "device": str(self.device)
        }
