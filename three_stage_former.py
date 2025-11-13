"""
Three-Stage Hierarchical Transformer for ECG Classification
Based on: Tang, Xiaoya, et al. "Hierarchical Transformer for Electrocardiogram Diagnosis"
(MIDL 2025)

This implementation provides a hierarchical transformer architecture with three stages
that process ECG signals at different temporal resolutions for multi-scale feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import math


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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StageTransformer(nn.Module):
    """Single stage transformer encoder."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        """
        Initialize a single stage transformer.
        
        Parameters:
        -----------
        d_model : int
            Model dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dim_feedforward : int
            Feedforward network dimension
        dropout : float
            Dropout rate
        """
        super(StageTransformer, self).__init__()
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stage transformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input of shape (seq_len, batch_size, d_model)
        
        Returns:
        --------
        torch.Tensor
            Output of same shape
        """
        return self.transformer(x)


class ThreeStageFormer(nn.Module):
    """
    Three-Stage Hierarchical Transformer for ECG Classification.
    
    The model processes ECG signals at three different temporal resolutions:
    - Stage 1: Fine-grained local patterns (high resolution)
    - Stage 2: Medium-scale patterns (medium resolution)
    - Stage 3: Coarse-grained global patterns (low resolution)
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers_per_stage: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 5,
        max_seq_len: int = 1000,
        pooling_stride: int = 2
    ):
        """
        Initialize Three-Stage Hierarchical Transformer.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        d_model : int
            Model embedding dimension
        nhead : int
            Number of attention heads
        num_layers_per_stage : int
            Number of transformer layers per stage
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout rate
        num_classes : int
            Number of classification classes
        max_seq_len : int
            Maximum sequence length
        pooling_stride : int
            Stride for pooling between stages
        """
        super(ThreeStageFormer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.pooling_stride = pooling_stride
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding for each stage
        self.pos_encoder_stage1 = PositionalEncoding(d_model, max_seq_len, dropout)
        self.pos_encoder_stage2 = PositionalEncoding(d_model, max_seq_len // pooling_stride, dropout)
        self.pos_encoder_stage3 = PositionalEncoding(d_model, max_seq_len // (pooling_stride ** 2), dropout)
        
        # Three-stage transformers
        self.stage1_transformer = StageTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_per_stage,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.stage2_transformer = StageTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_per_stage,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.stage3_transformer = StageTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers_per_stage,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Pooling layers for hierarchical downsampling
        self.pool1_to_2 = nn.AvgPool1d(kernel_size=pooling_stride, stride=pooling_stride)
        self.pool2_to_3 = nn.AvgPool1d(kernel_size=pooling_stride, stride=pooling_stride)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through three-stage hierarchical transformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals of shape (batch_size, seq_len, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initial embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # Stage 1: Fine-grained processing (full resolution)
        x1 = x.transpose(0, 1)  # (seq, batch, d_model)
        x1 = self.pos_encoder_stage1(x1)
        x1 = self.stage1_transformer(x1)  # (seq, batch, d_model)
        x1 = x1.transpose(0, 1)  # (batch, seq, d_model)
        
        # Pool for Stage 2
        x2_input = x1.transpose(1, 2)  # (batch, d_model, seq)
        x2_input = self.pool1_to_2(x2_input)  # (batch, d_model, seq/2)
        x2_input = x2_input.transpose(1, 2)  # (batch, seq/2, d_model)
        
        # Stage 2: Medium-scale processing
        x2 = x2_input.transpose(0, 1)  # (seq/2, batch, d_model)
        x2 = self.pos_encoder_stage2(x2)
        x2 = self.stage2_transformer(x2)  # (seq/2, batch, d_model)
        x2 = x2.transpose(0, 1)  # (batch, seq/2, d_model)
        
        # Pool for Stage 3
        x3_input = x2.transpose(1, 2)  # (batch, d_model, seq/2)
        x3_input = self.pool2_to_3(x3_input)  # (batch, d_model, seq/4)
        x3_input = x3_input.transpose(1, 2)  # (batch, seq/4, d_model)
        
        # Stage 3: Coarse-grained processing
        x3 = x3_input.transpose(0, 1)  # (seq/4, batch, d_model)
        x3 = self.pos_encoder_stage3(x3)
        x3 = self.stage3_transformer(x3)  # (seq/4, batch, d_model)
        x3 = x3.transpose(0, 1)  # (batch, seq/4, d_model)
        
        # Global pooling for each stage
        x1_pooled = self.global_pool(x1.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        x2_pooled = self.global_pool(x2.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        x3_pooled = self.global_pool(x3.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        
        # Feature fusion
        fused = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)  # (batch, d_model * 3)
        fused = self.fusion(fused)  # (batch, d_model)
        
        # Classification
        output = self.classifier(fused)  # (batch, num_classes)
        
        return output


class ECGDataset(Dataset):
    """Dataset class for ECG signals (reused from transformer_ecg.py)."""
    
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


def train_three_stage_former(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Train Three-Stage Former model.
    
    Parameters:
    -----------
    model : nn.Module
        Three-Stage Former model
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
    print("\nInitializing Three-Stage Former...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ThreeStageFormer(
        input_dim=1,
        d_model=128,
        nhead=8,
        num_layers_per_stage=2,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=5,
        max_seq_len=1000,
        pooling_stride=2
    )
    
    # Train model
    print("\nTraining Three-Stage Former model...")
    history = train_three_stage_former(
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

