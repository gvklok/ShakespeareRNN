"""
Training script for RNN text generator using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import re
import math

from .text_generator import LSTMTextGenerator, Tokenizer


class TextDataset(Dataset):
    """Dataset class for text sequences"""

    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class RNNTrainer:
    """Trainer class for RNN text generator"""

    def __init__(self, text_path: str = "data/training_text.txt",
                 device: str = None, max_vocab_size: int = 8000,
                 min_word_freq: int = 3):
        self.text_path = text_path
        self.tokenizer = Tokenizer(
            max_vocab_size=max_vocab_size,
            min_freq=min_word_freq
        )
        self.model = None
        self.max_sequence_len = 0
        self.total_words = 0
        self.X = None
        self.y = None
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq

        # Set device with MPS support for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device('cpu')
                print("Using CPU (no GPU available)")
        else:
            self.device = torch.device(device)
            print(f"Using device: {self.device}")

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^a-z0-9\s.,!?;:'\"\-]", '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def load_and_preprocess_data(self, max_seq_len: int = 50, sequence_stride: int = 1):
        """Load and preprocess training data

        Args:
            max_seq_len: Maximum sequence length to use (default: 50)
            sequence_stride: Step size for the sliding window (default: 1)
        """
        print("Loading training data...")
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Preprocess text
        text = self.preprocess_text(text)

        # Tokenization
        self.tokenizer.fit_on_texts([text])
        self.total_words = self.tokenizer.vocab_size

        print(f"Total unique words before filtering: {len(self.tokenizer.word_counts):,}")
        print(f"Using vocabulary size: {self.total_words:,}")
        print(f"Min word frequency: {self.min_word_freq}, Max vocab: {self.max_vocab_size or 'None'}")

        sequence_stride = max(1, sequence_stride)
        if sequence_stride > 1:
            print(f"Using sequence stride: {sequence_stride}")

        # Create input sequences using sliding window
        input_sequences = []
        skipped_unknown_targets = 0

        # Tokenize entire text
        token_list = self.tokenizer.texts_to_sequences([text])[0]

        # Create sequences using sliding window approach
        # This prevents memory issues with very long sequences
        for i in range(0, len(token_list) - max_seq_len, sequence_stride):
            seq = token_list[i:i + max_seq_len + 1]  # +1 for the target
            if seq[-1] == 0:
                skipped_unknown_targets += 1
                continue
            input_sequences.append(seq)

        if not input_sequences:
            raise ValueError("No training sequences remaining after filtering unknown targets.")

        print(f"Total sequences retained: {len(input_sequences)}")
        if skipped_unknown_targets:
            print(f"Skipped sequences with unknown target tokens: {skipped_unknown_targets}")

        # Convert to numpy array
        padded_sequences = np.array(input_sequences, dtype=np.int64)

        # Create predictors and labels
        self.X = padded_sequences[:, :-1]  # All but last token (input)
        self.y = padded_sequences[:, -1]   # Last token (target)

        self.max_sequence_len = max_seq_len + 1

        print(f"Max sequence length: {self.max_sequence_len}")
        print(f"Training data shape: X={self.X.shape}, y={self.y.shape}")

    def build_model(self, embedding_dim: int = 100, hidden_dim: int = 150,
                   num_layers: int = 2, dropout: float = 0.2):
        """Build the RNN model"""
        self.model = LSTMTextGenerator(
            vocab_size=self.total_words,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=0
        )

        self.model.to(self.device)

        # Print model summary
        print("\nModel Architecture:")
        print(f"{'='*60}")
        print(self.model)
        print(f"{'='*60}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")

        # Store model config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def train(self, epochs: int = 100, batch_size: int = 128,
             validation_split: float = 0.1, learning_rate: float = 0.001,
             patience: int = 20):
        """Train the model

        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            learning_rate: Learning rate for optimizer
            patience: Number of epochs to wait before early stopping
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Create data loaders
        dataset_size = len(self.X)
        val_size = max(1, int(dataset_size * validation_split))
        train_size = dataset_size - val_size
        if train_size <= 0:
            raise ValueError("Validation split leaves no data for training. Reduce validation_split.")

        # Split data
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = TextDataset(self.X[train_indices], self.y[train_indices])
        val_dataset = TextDataset(self.X[val_indices], self.y[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=1e-4
        )

        # Setup tensorboard
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f'visualizations/logs/{timestamp}')

        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'perplexity': [],
            'val_perplexity': [],
            'grad_norm': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting training...")
        print(f"{'='*60}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            grad_norm_sum = 0.0
            grad_steps = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for sequences, labels in train_pbar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs, _ = self.model(sequences)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=3.0
                )
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.detach().item()
                grad_norm_sum += grad_norm
                grad_steps += 1
                optimizer.step()

                # Statistics
                train_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })

            # Calculate average training metrics
            avg_train_loss = train_loss / train_total
            avg_train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ')
                for sequences, labels in val_pbar:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)

                    outputs, _ = self.model(sequences)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * sequences.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * val_correct / val_total:.2f}%'
                    })

            # Calculate average validation metrics
            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total

            avg_grad_norm = grad_norm_sum / max(1, grad_steps)
            train_perplexity = math.exp(min(avg_train_loss, 10))
            val_perplexity = math.exp(min(avg_val_loss, 10))

            # Scheduler step for next epoch
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(avg_train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_acc)
            history['perplexity'].append(train_perplexity)
            history['val_perplexity'].append(val_perplexity)
            history['grad_norm'].append(avg_grad_norm)
            history['learning_rate'].append(current_lr)

            # Log to tensorboard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
            writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
            writer.add_scalar('Perplexity/train', train_perplexity, epoch)
            writer.add_scalar('Perplexity/val', val_perplexity, epoch)
            writer.add_scalar('GradNorm/train', avg_grad_norm, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc*100:.2f}%")
            print(f"  Train PPL: {train_perplexity:.2f}, Val PPL: {val_perplexity:.2f}")
            print(f"  Avg Grad Norm: {avg_grad_norm:.2f}, LR: {current_lr:.6f}")

            # Early stopping and model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best model
                self._save_checkpoint(
                    epoch=epoch,
                    model_path='saved_models/model.pt',
                    optimizer=optimizer,
                    loss=avg_val_loss
                )
                print(f"  ✓ Model saved (best val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Early stopping patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

            print(f"{'='*60}")

        writer.close()

        # Save final training history
        with open('saved_models/training_history.pkl', 'wb') as f:
            pickle.dump(history, f)

        # Save tokenizer and metadata
        with open('saved_models/tokenizer.pkl', 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'max_sequence_len': self.max_sequence_len,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'max_vocab_size': self.max_vocab_size,
                'min_word_freq': self.min_word_freq
            }, f)

        # Plot training metrics
        self._plot_training_history(history)

        print("\n✓ Training complete!")
        return history

    def _save_checkpoint(self, epoch, model_path, optimizer, loss):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

    def _plot_training_history(self, history):
        """Plot and save training metrics"""
        os.makedirs('visualizations', exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot([acc * 100 for acc in history['accuracy']],
                     label='Training Accuracy', linewidth=2)
        axes[1].plot([acc * 100 for acc in history['val_accuracy']],
                     label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Perplexity plot
        axes[2].plot(history['perplexity'], label='Training Perplexity', linewidth=2)
        axes[2].plot(history['val_perplexity'], label='Validation Perplexity', linewidth=2)
        axes[2].set_title('Model Perplexity', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Perplexity', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/training_history.png', dpi=300, bbox_inches='tight')
        print("Training plots saved to visualizations/training_history.png")
        plt.close()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    print("="*70)
    print("RNN Text Generator - IMPROVED MODEL V2")
    print("="*70)
    print("\nImproved Configuration:")
    print("  • Vocabulary: 5,000 words (richer generation)")
    print("  • Sequences: ~440K (stride=2)")
    print("  • Sequence length: 50 words (guide recommendation)")
    print("  • LSTM units: 256 (proven capacity)")
    print("  • Layers: 2")
    print("  • Embedding dim: 128")
    print("  • Dropout: 0.15 (less regularization)")
    print("  • Batch size: 128")
    print("  • Learning rate: 0.001")
    print("  • Patience: 12 epochs (train longer)")
    print("\n  Goal: Better coherence & lower perplexity")
    print("="*70)

    # Initialize trainer with larger vocabulary
    print("\n📚 Initializing trainer...")
    trainer = RNNTrainer(max_vocab_size=5000, min_word_freq=2)

    # Load with stride for faster training
    print("\nLoading Complete Works with 50-word sequences (every 2nd for balance)...")
    trainer.load_and_preprocess_data(max_seq_len=50, sequence_stride=2)

    # Build model with less dropout
    print("\n🏗️  Building model with 256 LSTM units...")
    trainer.build_model(
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.15
    )

    # Train with longer patience
    print("\n🚀 Starting training...")
    trainer.train(
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
        patience=12
    )
