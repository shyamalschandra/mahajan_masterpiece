"""
CNN and LSTM-based ECG Classification Models
These implementations provide alternative deep learning approaches for ECG classification:
1. 1D Convolutional Neural Network (CNN) - Extracts local patterns using convolution
2. Long Short-Term Memory (LSTM) - Sequential modeling with recurrent connections

Both models process raw ECG signals and are commonly used in ECG analysis literature.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import math


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
        if len(self.signals.shape) == 2:
            self.signals = self.signals[:, :, np.newaxis]
    
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
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # Padding or truncation
        if signal.shape[0] < self.seq_len:
            pad_len = self.seq_len - signal.shape[0]
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode='constant')
        elif signal.shape[0] > self.seq_len:
            signal = signal[:self.seq_len]
        
        # Convert to torch tensors
        signal = torch.FloatTensor(signal)
        label = torch.LongTensor([label])
        
        return signal, label


class CNN1DECGClassifier(nn.Module):
    """
    1D Convolutional Neural Network for ECG Classification.
    
    Uses 1D convolutions to extract local temporal patterns from ECG signals.
    This architecture is commonly used for ECG analysis and is effective at
    capturing morphological features like QRS complexes, P waves, and T waves.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 5,
        seq_len: int = 1000,
        dropout: float = 0.3
    ):
        """
        Initialize 1D CNN ECG Classifier.
        
        Parameters:
        -----------
        input_channels : int
            Number of input channels (1 for single-lead ECG)
        num_classes : int
            Number of classification classes
        seq_len : int
            Sequence length
        dropout : float
            Dropout rate
        """
        super(CNN1DECGClassifier, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size after convolutions and pooling
        # After 4 maxpool layers with stride=2: seq_len / 16
        flattened_size = (seq_len // 16) * 256
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D CNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals of shape (batch_size, seq_len, input_channels)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        # Reshape: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # Classification
        x = self.classifier(x)
        
        return x


class LSTMECGClassifier(nn.Module):
    """
    Long Short-Term Memory (LSTM) Network for ECG Classification.
    
    Uses LSTM cells to model sequential dependencies in ECG signals.
    LSTMs are effective at capturing temporal patterns and long-range
    dependencies through their recurrent connections and gating mechanisms.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM ECG Classifier.
        
        Parameters:
        -----------
        input_size : int
            Input feature dimension (1 for single-lead ECG)
        hidden_size : int
            Hidden state dimension
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of classification classes
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTMECGClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size (doubled if bidirectional)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals of shape (batch_size, seq_len, input_size)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # Output shape: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state (or concatenate forward and backward if bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            forward_hidden = h_n[-2, :, :]  # Last forward layer
            backward_hidden = h_n[-1, :, :]  # Last backward layer
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = h_n[-1, :, :]  # Last layer hidden state
        
        # Classification
        output = self.classifier(final_hidden)
        
        return output


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Train CNN or LSTM model.
    
    Parameters:
    -----------
    model : nn.Module
        CNN or LSTM model
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
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.squeeze().to(device)
        
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals = signals.to(device)
                    labels = labels.squeeze().to(device)
                    
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}"
            if val_loader is not None:
                msg += f" - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}"
            print(msg)
    
    return history


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
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
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
    np.random.seed(42)
    signals = []
    labels = []
    
    for i in range(n_samples):
        class_id = i % num_classes
        
        # Generate synthetic ECG-like signal
        t = np.linspace(0, 2 * np.pi, seq_len)
        
        # Base ECG pattern with variations by class
        if class_id == 0:  # Normal
            signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(3 * t)
        elif class_id == 1:  # Atrial Premature Contraction (APC)
            signal = np.sin(t) + 0.8 * np.sin(1.5 * t) + 0.2 * np.sin(4 * t)
        elif class_id == 2:  # Ventricular Premature Contraction (VPC)
            signal = 1.2 * np.sin(0.8 * t) + 0.6 * np.sin(2.5 * t) + 0.4 * np.sin(5 * t)
        elif class_id == 3:  # Fusion
            signal = 0.8 * np.sin(t) + 0.7 * np.sin(1.8 * t) + 0.3 * np.sin(3.5 * t)
        else:  # Other
            signal = np.sin(1.2 * t) + 0.6 * np.sin(2.2 * t) + 0.4 * np.sin(4.5 * t)
        
        # Add noise
        signal += noise_level * np.random.randn(seq_len)
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        signals.append(signal)
        labels.append(class_id)
    
    return np.array(signals), np.array(labels)


if __name__ == "__main__":
    # Example usage for CNN
    print("Creating synthetic ECG dataset...")
    signals, labels = create_synthetic_ecg_data(
        n_samples=2000,
        seq_len=1000,
        num_classes=5,
        noise_level=0.1
    )
    
    # Split data
    split1 = int(0.7 * len(signals))
    split2 = int(0.85 * len(signals))
    
    X_train, y_train = signals[:split1], labels[:split1]
    X_val, y_val = signals[split1:split2], labels[split1:split2]
    X_test, y_test = signals[split2:], labels[split2:]
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train, seq_len=1000)
    val_dataset = ECGDataset(X_val, y_val, seq_len=1000)
    test_dataset = ECGDataset(X_test, y_test, seq_len=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test CNN
    print("\n" + "="*60)
    print("Testing 1D CNN Model")
    print("="*60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cnn_model = CNN1DECGClassifier(
        input_channels=1,
        num_classes=5,
        seq_len=1000,
        dropout=0.3
    )
    
    print("\nTraining CNN model...")
    history = train_model(
        cnn_model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=True
    )
    
    print("\nEvaluating CNN on test set...")
    results = evaluate_model(cnn_model, test_loader, device=device)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Test LSTM
    print("\n" + "="*60)
    print("Testing LSTM Model")
    print("="*60)
    
    lstm_model = LSTMECGClassifier(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
        bidirectional=True
    )
    
    print("\nTraining LSTM model...")
    history = train_model(
        lstm_model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=True
    )
    
    print("\nEvaluating LSTM on test set...")
    results = evaluate_model(lstm_model, test_loader, device=device)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")

