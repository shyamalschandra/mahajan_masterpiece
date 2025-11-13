"""
Transformer-based ECG Classification for Early Detection of Cardiac Arrhythmias
Based on: Ikram, Sunnia, et al. "Transformer-based ECG classification for early 
detection of cardiac arrhythmias." Frontiers in Medicine 12 (2025): 1600855.

This implementation provides a Transformer architecture optimized for ECG signal classification.
"""

import numpy as np                                                      # NumPy for array operations
import torch                                                            # PyTorch for deep learning
import torch.nn as nn                                                   # Neural network modules
import torch.optim as optim                                             # Optimizers
from torch.utils.data import Dataset, DataLoader                        # Data loading utilities
from typing import Tuple, Optional                                      # Type hints
import math                                                             # Math functions


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer to capture temporal information."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Parameters:
        -----------
        d_model : int
            Dimension of model embeddings
        max_len : int
            Maximum sequence length
        dropout : float
            Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)                              # Initialize PE matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Position indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *      # Division term for sin/cos
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)                     # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)                     # Odd indices: cos
        pe = pe.unsqueeze(0).transpose(0, 1)                             # Reshape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)                                   # Register as buffer (not parameter)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, d_model)
        
        Returns:
        --------
        torch.Tensor
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]                                  # Add positional encoding
        return self.dropout(x)                                          # Apply dropout


class TransformerECGClassifier(nn.Module):
    """
    Transformer-based model for ECG signal classification.
    Implements encoder-only Transformer architecture optimized for time series.
    """
    
    def __init__(
        self,
        input_dim: int = 1,                                              # Input feature dimension (single lead)
        d_model: int = 128,                                              # Model dimension
        nhead: int = 8,                                                  # Number of attention heads
        num_layers: int = 6,                                             # Number of transformer layers
        dim_feedforward: int = 512,                                      # Feedforward network dimension
        dropout: float = 0.1,                                            # Dropout rate
        num_classes: int = 5,                                             # Number of output classes
        max_seq_len: int = 1000                                          # Maximum sequence length
    ):
        """
        Initialize Transformer ECG Classifier.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        d_model : int
            Model embedding dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout rate
        num_classes : int
            Number of classification classes
        max_seq_len : int
            Maximum sequence length
        """
        super(TransformerECGClassifier, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)            # Linear projection to d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout) # Add temporal information
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,                                             # Model dimension
            nhead=nhead,                                                 # Multi-head attention
            dim_feedforward=dim_feedforward,                            # FFN dimension
            dropout=dropout,                                              # Dropout rate
            activation='gelu',                                           # GELU activation
            batch_first=False                                             # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,                                               # Encoder layer definition
            num_layers=num_layers                                         # Stack multiple layers
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)                       # Global average pooling
        self.classifier = nn.Sequential(                                 # Classification layers
            nn.Linear(d_model, dim_feedforward // 2),                   # First linear layer
            nn.ReLU(),                                                   # ReLU activation
            nn.Dropout(dropout),                                         # Dropout
            nn.Linear(dim_feedforward // 2, num_classes)                 # Output layer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals of shape (batch_size, seq_len, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape                                 # Get dimensions
        
        # Embedding layer
        x = self.input_embedding(x)                                      # Project to d_model: (batch, seq, d_model)
        x = x.transpose(0, 1)                                            # Transpose: (seq, batch, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)                                          # Add temporal position info
        
        # Transformer encoding
        x = self.transformer_encoder(x)                                  # Apply transformer layers
        
        # Global pooling and classification
        x = x.transpose(0, 1)                                            # Transpose back: (batch, seq, d_model)
        x = x.transpose(1, 2)                                            # For pooling: (batch, d_model, seq)
        x = self.global_pool(x)                                          # Global average pooling: (batch, d_model, 1)
        x = x.squeeze(-1)                                                # Remove last dim: (batch, d_model)
        x = self.classifier(x)                                           # Classification: (batch, num_classes)
        
        return x                                                         # Return logits


class ECGDataset(Dataset):
    """Dataset class for ECG signals."""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray, seq_len: int = 1000):
        """
        Initialize ECG dataset.
        
        Parameters:
        -----------
        signals : np.ndarray
            ECG signals of shape (n_samples, seq_len) or (n_samples, seq_len, features)
        labels : np.ndarray
            Class labels of shape (n_samples,)
        seq_len : int
            Sequence length to use (padding/truncation)
        """
        self.signals = signals
        self.labels = labels
        self.seq_len = seq_len
        
        # Ensure signals are 2D (n_samples, seq_len, features)
        if len(self.signals.shape) == 2:                                # If 2D, add feature dimension
            self.signals = self.signals[:, :, np.newaxis]                # (n_samples, seq_len, 1)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Parameters:
        -----------
        idx : int
            Sample index
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (signal, label) pair
        """
        signal = self.signals[idx]                                       # Get signal
        label = self.labels[idx]                                         # Get label
        
        # Padding or truncation
        if signal.shape[0] < self.seq_len:                              # If shorter than seq_len
            pad_len = self.seq_len - signal.shape[0]                     # Calculate padding length
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode='constant') # Pad with zeros
        elif signal.shape[0] > self.seq_len:                             # If longer than seq_len
            signal = signal[:self.seq_len]                               # Truncate
        
        # Convert to torch tensors
        signal = torch.FloatTensor(signal)                              # Convert to float tensor
        label = torch.LongTensor([label])                                # Convert to long tensor
        
        return signal, label                                             # Return (signal, label)


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Train Transformer model.
    
    Parameters:
    -----------
    model : nn.Module
        Transformer model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict
        Training history
    """
    model = model.to(device)                                             # Move model to device
    criterion = nn.CrossEntropyLoss()                                   # Cross-entropy loss for multi-class
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)      # AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False    # Learning rate scheduler
    )
    
    history = {
        'train_loss': [],                                                # Training loss history
        'train_acc': [],                                                 # Training accuracy history
        'val_loss': [],                                                  # Validation loss history
        'val_acc': []                                                    # Validation accuracy history
    }
    
    best_val_loss = float('inf')                                         # Track best validation loss
    patience = 10                                                        # Early stopping patience
    patience_counter = 0                                                 # Patience counter
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()                                                    # Set to training mode
        train_loss = 0.0                                                # Accumulate training loss
        train_correct = 0                                                # Count correct predictions
        train_total = 0                                                  # Count total samples
        
        for signals, labels in train_loader:
            signals = signals.to(device)                                # Move to device
            labels = labels.squeeze().to(device)                         # Move to device and squeeze
        
            optimizer.zero_grad()                                         # Zero gradients
            outputs = model(signals)                                     # Forward pass
            loss = criterion(outputs, labels)                            # Compute loss
            loss.backward()                                              # Backward pass
            optimizer.step()                                             # Update weights
            
            train_loss += loss.item()                                    # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)                    # Get predictions
            train_total += labels.size(0)                                # Count samples
            train_correct += (predicted == labels).sum().item()          # Count correct
        
        avg_train_loss = train_loss / len(train_loader)                  # Average training loss
        train_acc = train_correct / train_total                          # Training accuracy
        
        history['train_loss'].append(avg_train_loss)                     # Store training loss
        history['train_acc'].append(train_acc)                           # Store training accuracy
        
        # Validation phase
        if val_loader is not None:
            model.eval()                                                 # Set to evaluation mode
            val_loss = 0.0                                               # Accumulate validation loss
            val_correct = 0                                              # Count correct predictions
            val_total = 0                                                # Count total samples
            
            with torch.no_grad():                                        # Disable gradient computation
                for signals, labels in val_loader:
                    signals = signals.to(device)                        # Move to device
                    labels = labels.squeeze().to(device)                 # Move to device and squeeze
                    
                    outputs = model(signals)                             # Forward pass
                    loss = criterion(outputs, labels)                   # Compute loss
                    
                    val_loss += loss.item()                              # Accumulate loss
                    _, predicted = torch.max(outputs.data, 1)           # Get predictions
                    val_total += labels.size(0)                          # Count samples
                    val_correct += (predicted == labels).sum().item()    # Count correct
            
            avg_val_loss = val_loss / len(val_loader)                    # Average validation loss
            val_acc = val_correct / val_total                            # Validation accuracy
            
            history['val_loss'].append(avg_val_loss)                     # Store validation loss
            history['val_acc'].append(val_acc)                           # Store validation accuracy
            
            scheduler.step(avg_val_loss)                                 # Update learning rate
            
            # Early stopping
            if avg_val_loss < best_val_loss:                             # Check if improved
                best_val_loss = avg_val_loss                             # Update best loss
                patience_counter = 0                                     # Reset patience
            else:
                patience_counter += 1                                   # Increment patience
                if patience_counter >= patience:                         # Check if exceeded
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break                                                # Stop training
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}"
            if val_loader is not None:
                msg += f" - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}"
            print(msg)
    
    return history                                                       # Return training history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate model on test set.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : str
        Device to evaluate on
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    model = model.to(device)                                             # Move model to device
    model.eval()                                                         # Set to evaluation mode
    
    all_predictions = []                                                 # Store all predictions
    all_labels = []                                                      # Store all labels
    correct = 0                                                          # Count correct predictions
    total = 0                                                            # Count total samples
    
    criterion = nn.CrossEntropyLoss()                                   # Loss function
    total_loss = 0.0                                                     # Accumulate loss
    
    with torch.no_grad():                                                # Disable gradient computation
        for signals, labels in test_loader:
            signals = signals.to(device)                                # Move to device
            labels = labels.squeeze().to(device)                         # Move to device and squeeze
            
            outputs = model(signals)                                     # Forward pass
            loss = criterion(outputs, labels)                            # Compute loss
            total_loss += loss.item()                                    # Accumulate loss
            
            _, predicted = torch.max(outputs.data, 1)                    # Get predictions
            total += labels.size(0)                                      # Count samples
            correct += (predicted == labels).sum().item()                # Count correct
            
            all_predictions.extend(predicted.cpu().numpy())            # Store predictions
            all_labels.extend(labels.cpu().numpy())                      # Store labels
    
    accuracy = correct / total                                           # Calculate accuracy
    avg_loss = total_loss / len(test_loader)                             # Average loss
    
    return {
        'accuracy': accuracy,                                             # Accuracy score
        'loss': avg_loss,                                                # Average loss
        'predictions': np.array(all_predictions),                        # All predictions
        'labels': np.array(all_labels)                                    # All labels
    }


def create_synthetic_ecg_data(
    n_samples: int = 1000,
    seq_len: int = 1000,
    num_classes: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic ECG-like signals for demonstration.
    In practice, replace with real MIT-BIH or other ECG datasets.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    seq_len : int
        Sequence length
    num_classes : int
        Number of classes
    noise_level : float
        Noise level
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (signals, labels)
    """
    np.random.seed(42)                                                   # Set seed for reproducibility
    signals = []                                                         # List to store signals
    labels = []                                                          # List to store labels
    
    for i in range(n_samples):
        class_id = i % num_classes                                       # Assign class cyclically
        
        # Generate synthetic ECG-like signal
        t = np.linspace(0, 2 * np.pi, seq_len)                          # Time axis
        
        # Base ECG pattern with variations by class
        if class_id == 0:  # Normal
            signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(3 * t) # Normal rhythm
        elif class_id == 1:  # Atrial Premature Contraction (APC)
            signal = np.sin(t) + 0.8 * np.sin(1.5 * t) + 0.2 * np.sin(4 * t) # Irregular pattern
        elif class_id == 2:  # Ventricular Premature Contraction (VPC)
            signal = 1.2 * np.sin(0.8 * t) + 0.6 * np.sin(2.5 * t) + 0.4 * np.sin(5 * t) # Abnormal
        elif class_id == 3:  # Fusion
            signal = 0.8 * np.sin(t) + 0.7 * np.sin(1.8 * t) + 0.3 * np.sin(3.5 * t) # Mixed
        else:  # Other
            signal = np.sin(1.2 * t) + 0.6 * np.sin(2.2 * t) + 0.4 * np.sin(4.5 * t) # Variant
        
        # Add noise
        signal += noise_level * np.random.randn(seq_len)                # Add Gaussian noise
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)       # Normalize signal
        
        signals.append(signal)                                           # Append signal
        labels.append(class_id)                                         # Append label
    
    return np.array(signals), np.array(labels)                           # Return as arrays


if __name__ == "__main__":
    # Example usage
    print("Creating synthetic ECG dataset...")
    signals, labels = create_synthetic_ecg_data(
        n_samples=2000,                                                 # 2000 samples
        seq_len=1000,                                                    # 1000 timesteps
        num_classes=5,                                                   # 5 classes
        noise_level=0.1                                                  # 10% noise
    )
    
    # Split data
    split1 = int(0.7 * len(signals))                                     # 70% train
    split2 = int(0.85 * len(signals))                                    # 15% val, 15% test
    
    X_train, y_train = signals[:split1], labels[:split1]                # Training set
    X_val, y_val = signals[split1:split2], labels[split1:split2]         # Validation set
    X_test, y_test = signals[split2:], labels[split2:]                   # Test set
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)          # Training dataset
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)                 # Validation dataset
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)             # Test dataset
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Training loader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)   # Validation loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Test loader
    
    # Initialize model
    print("\nInitializing Transformer ECG Classifier...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'              # Use GPU if available
    print(f"Using device: {device}")
    
    model = TransformerECGClassifier(
        input_dim=1,                                                     # Single channel ECG
        d_model=128,                                                     # Model dimension
        nhead=8,                                                          # 8 attention heads
        num_layers=6,                                                     # 6 transformer layers
        dim_feedforward=512,                                             # FFN dimension
        dropout=0.1,                                                     # 10% dropout
        num_classes=5,                                                    # 5 classes
        max_seq_len=1000                                                  # Max sequence length
    )
    
    # Train model
    print("\nTraining Transformer model...")
    history = train_transformer(
        model, train_loader, val_loader,
        num_epochs=50,                                                   # 50 epochs
        learning_rate=0.001,                                              # Learning rate
        device=device,                                                    # Device
        verbose=True                                                      # Print progress
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=device)          # Evaluate model
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"\nClass distribution in predictions: {np.bincount(results['predictions'])}")
    print(f"Class distribution in labels: {np.bincount(results['labels'])}")

