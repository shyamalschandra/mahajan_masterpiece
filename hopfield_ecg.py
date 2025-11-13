"""
Hopfield Network for ECG Classification
Based on: Hopfield Network applications in ECG analysis
(ETASR, 2013)

Hopfield Networks are energy-based recurrent neural networks that can be used
for pattern recognition and associative memory. They are particularly useful
for ECG classification as they can store and recall patterns efficiently.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import time


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


class HopfieldNetwork(nn.Module):
    """
    Hopfield Network for ECG Classification.
    
    Hopfield Networks are energy-based recurrent neural networks that can
    store and recall patterns. They use an energy function to converge to
    stable states representing stored patterns.
    
    This implementation uses a modern continuous Hopfield network approach
    with learnable weights for ECG classification.
    """
    
    def __init__(
        self,
        input_size: int = 1000,
        hidden_size: int = 256,
        num_classes: int = 5,
        num_iterations: int = 10,
        beta: float = 1.0
    ):
        """
        Initialize Hopfield Network.
        
        Parameters:
        -----------
        input_size : int
            Size of input pattern (sequence length)
        hidden_size : int
            Size of hidden/associative layer
        num_classes : int
            Number of classification classes
        num_iterations : int
            Number of iterations for pattern convergence
        beta : float
            Inverse temperature parameter (controls sharpness of activation)
        """
        super(HopfieldNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.beta = beta
        
        # Input projection to hidden space
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Hopfield weight matrix (symmetric, stores patterns)
        # Using learnable weights instead of Hebbian learning
        self.hopfield_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )
        # Make symmetric (Hopfield networks require symmetric weights)
        with torch.no_grad():
            self.hopfield_weights.data = (
                self.hopfield_weights.data + self.hopfield_weights.data.T
            ) / 2
        
        # Bias for hidden units
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def hopfield_update(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one Hopfield network update step.
        
        Parameters:
        -----------
        x : torch.Tensor
            Current state of hidden units (batch_size, hidden_size)
        
        Returns:
        --------
        torch.Tensor
            Updated state
        """
        # Energy-based update: x_new = tanh(beta * (W @ x + b))
        # This is the continuous Hopfield network update rule
        energy = torch.matmul(x, self.hopfield_weights) + self.hidden_bias
        x_new = torch.tanh(self.beta * energy)
        return x_new
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Hopfield Network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals of shape (batch_size, seq_len, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Flatten input: (batch, seq_len, features) -> (batch, seq_len)
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove feature dimension if present
        
        # Project input to hidden space
        h = self.input_projection(x)  # (batch, hidden_size)
        
        # Initialize with projected input
        h_current = torch.tanh(h)
        
        # Iterative Hopfield updates (pattern convergence)
        for _ in range(self.num_iterations):
            h_current = self.hopfield_update(h_current)
        
        # Use converged state for classification
        output = self.classifier(h_current)
        
        return output


class HopfieldECGClassifier(nn.Module):
    """
    Enhanced Hopfield Network Classifier for ECG.
    
    This version includes preprocessing and feature extraction
    before the Hopfield network, making it more suitable for
    ECG classification tasks.
    """
    
    def __init__(
        self,
        input_size: int = 1000,
        feature_size: int = 128,
        hidden_size: int = 256,
        num_classes: int = 5,
        num_iterations: int = 10,
        beta: float = 1.0
    ):
        """
        Initialize Enhanced Hopfield Network.
        
        Parameters:
        -----------
        input_size : int
            Input sequence length
        feature_size : int
            Size of feature representation
        hidden_size : int
            Size of Hopfield associative layer
        num_classes : int
            Number of classes
        num_iterations : int
            Number of Hopfield iterations
        beta : float
            Inverse temperature parameter
        """
        super(HopfieldECGClassifier, self).__init__()
        
        # Feature extraction layer (reduces dimensionality)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, feature_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU()
        )
        
        # Hopfield Network
        self.hopfield_net = HopfieldNetwork(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_iterations=num_iterations,
            beta=beta
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals (batch_size, seq_len, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Classification logits
        """
        # Flatten if needed
        if x.dim() == 3:
            x = x.squeeze(-1)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch, feature_size)
        
        # Reshape for Hopfield network (add dimension)
        features = features.unsqueeze(-1)  # (batch, feature_size, 1)
        
        # Pass through Hopfield network
        output = self.hopfield_net(features)
        
        return output


def train_hopfield(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Train Hopfield Network model.
    
    Parameters:
    -----------
    model : nn.Module
        Hopfield Network model
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
            
            # Ensure Hopfield weights remain symmetric after gradient update
            with torch.no_grad():
                if hasattr(model, 'hopfield_net'):
                    model.hopfield_net.hopfield_weights.data = (
                        model.hopfield_net.hopfield_weights.data + 
                        model.hopfield_net.hopfield_weights.data.T
                    ) / 2
            
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
    # Example usage
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
    
    # Initialize model
    print("\nInitializing Hopfield Network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = HopfieldECGClassifier(
        input_size=1000,
        feature_size=128,
        hidden_size=256,
        num_classes=5,
        num_iterations=10,
        beta=1.0
    )
    
    # Train model
    print("\nTraining Hopfield Network model...")
    history = train_hopfield(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=device)
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"\nClass distribution in predictions: {np.bincount(results['predictions'])}")
    print(f"Class distribution in labels: {np.bincount(results['labels'])}")

