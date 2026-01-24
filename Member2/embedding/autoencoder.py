"""
Autoencoder Embedding Model
Neural network autoencoder for learning nonlinear embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
import json
from pathlib import Path
from tqdm import tqdm

from .base import BaseEmbeddingModel


class AutoencoderNetwork(nn.Module):
    """PyTorch autoencoder network."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        encoder_layers: Tuple[int, ...] = (128, 64),
        activation: str = 'relu',
        dropout_rate: float = 0.2
    ):
        """
        Initialize autoencoder architecture.

        Parameters
        ----------
        input_dim : int
            Input dimension (e.g., 170)
        embedding_dim : int
            Bottleneck/embedding dimension (e.g., 32)
        encoder_layers : tuple
            Hidden layer sizes for encoder
        activation : str
            Activation function ('relu', 'leaky_relu', 'elu')
        dropout_rate : float
            Dropout rate for regularization
        """
        super(AutoencoderNetwork, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Choose activation
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            act = nn.ELU()
        else:
            act = nn.ReLU()

        # Build encoder
        encoder_modules = []
        prev_dim = input_dim

        for hidden_dim in encoder_layers:
            encoder_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                act,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Bottleneck layer (no activation, no dropout)
        encoder_modules.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder (mirror of encoder)
        decoder_modules = []
        prev_dim = embedding_dim

        for hidden_dim in reversed(encoder_layers):
            decoder_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                act,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        decoder_modules.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from encoder."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from embeddings."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed input
        embeddings : torch.Tensor
            Bottleneck embeddings
        """
        embeddings = self.encode(x)
        reconstructed = self.decode(embeddings)
        return reconstructed, embeddings


class AutoencoderModel(BaseEmbeddingModel):
    """Autoencoder-based embedding model."""

    def __init__(
        self,
        input_dim: int = 170,
        embedding_dim: int = 32,
        encoder_layers: Tuple[int, ...] = (128, 64),
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[str] = None
    ):
        """
        Initialize autoencoder model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        embedding_dim : int
            Target embedding dimension
        encoder_layers : tuple
            Hidden layer sizes
        activation : str
            Activation function
        dropout_rate : float
            Dropout rate
        learning_rate : float
            Learning rate for Adam optimizer
        batch_size : int
            Training batch size
        epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Early stopping patience
        device : str, optional
            Device ('cpu', 'cuda', 'mps'). Auto-detect if None.
        """
        super().__init__(input_dim, embedding_dim, name="Autoencoder")

        self.encoder_layers = encoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Initialize network
        self.network = AutoencoderNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            encoder_layers=encoder_layers,
            activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def fit(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'AutoencoderModel':
        """
        Train the autoencoder.

        Parameters
        ----------
        X : np.ndarray
            Training data (N x input_dim)
        X_val : np.ndarray, optional
            Validation data
        verbose : bool
            Print training progress

        Returns
        -------
        self
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation loader
        val_loader = None
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Optimizer and loss
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        # Training loop
        if verbose:
            print(f"\n🔧 Training Autoencoder on {self.device}")
            print(f"   Architecture: {self.input_dim} → {' → '.join(map(str, self.encoder_layers))} → {self.embedding_dim}")
            print(f"   Training samples: {len(X)}")
            if X_val is not None:
                print(f"   Validation samples: {len(X_val)}")
            print("=" * 60)

        epoch_iterator = range(self.epochs)
        if verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training")

        for epoch in epoch_iterator:
            # Training phase
            self.network.train()
            train_loss = 0.0

            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self.network(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validation phase
            if val_loader is not None:
                self.network.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        reconstructed, _ = self.network(batch_x)
                        loss = criterion(reconstructed, batch_x)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.history['val_loss'].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.network.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break

                # Update progress bar
                if verbose and hasattr(epoch_iterator, 'set_postfix'):
                    epoch_iterator.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}'
                    })
            else:
                if verbose and hasattr(epoch_iterator, 'set_postfix'):
                    epoch_iterator.set_postfix({'train_loss': f'{train_loss:.4f}'})

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)
            if verbose:
                print(f"\n✓ Restored best model (val_loss: {best_val_loss:.4f})")

        self.is_fitted = True

        if verbose:
            print(f"✅ Training complete!")
            print(f"   Final train loss: {self.history['train_loss'][-1]:.4f}")
            if self.history['val_loss']:
                print(f"   Final val loss: {self.history['val_loss'][-1]:.4f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to embedding space.

        Parameters
        ----------
        X : np.ndarray
            Input data (N x input_dim)

        Returns
        -------
        embeddings : np.ndarray
            Embeddings (N x embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        self.network.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            embeddings = self.network.encode(X_tensor)
            embeddings_np = embeddings.cpu().numpy()

        return embeddings_np

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input from embeddings.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        reconstructed : np.ndarray
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reconstruct")

        self.network.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed, _ = self.network(X_tensor)
            reconstructed_np = reconstructed.cpu().numpy()

        return reconstructed_np

    def save(self, filepath: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.network.state_dict(), str(filepath))

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'encoder_layers': list(self.encoder_layers),
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'history': self.history
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'AutoencoderModel':
        """Load model from disk."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            input_dim=metadata['input_dim'],
            embedding_dim=metadata['embedding_dim'],
            encoder_layers=tuple(metadata['encoder_layers']),
            activation=metadata['activation'],
            dropout_rate=metadata['dropout_rate'],
            learning_rate=metadata['learning_rate']
        )

        # Load weights
        model.network.load_state_dict(torch.load(str(filepath), map_location=model.device))
        model.history = metadata.get('history', {'train_loss': [], 'val_loss': []})
        model.is_fitted = True

        print(f"✓ Model loaded from {filepath}")
        return model

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)

        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Autoencoder Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.show()


if __name__ == '__main__':
    # Test autoencoder
    print("Testing Autoencoder Model...\n")

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(500, 170).astype(np.float32)
    X_val = np.random.randn(100, 170).astype(np.float32)

    # Create and train model
    model = AutoencoderModel(
        input_dim=170,
        embedding_dim=32,
        encoder_layers=(128, 64),
        epochs=10,
        batch_size=32,
        device='cpu'
    )

    print("Training model...")
    model.fit(X_train, X_val, verbose=True)

    # Transform
    embeddings = model.transform(X_train[:10])
    print(f"\n✓ Embeddings shape: {embeddings.shape}")

    # Reconstruct
    reconstructed = model.reconstruct(X_train[:10])
    mse = np.mean((X_train[:10] - reconstructed) ** 2)
    print(f"✓ Reconstruction MSE: {mse:.4f}")

    print("\n✅ Autoencoder test complete")
