"""
Variational Autoencoder (VAE) for ECG Classification
Based on: FactorECG - Explainable AI for ECG analysis
(van de Leur et al., 2022, European Heart Journal - Digital Health)

The VAE compresses ECG signals into latent factors that can be used
for both reconstruction and classification tasks, providing explainability
through interpretable latent representations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class VAEEcgEncoder(nn.Module):
    """
    Encoder network for VAE.
    Maps ECG signals to latent space (mean and log variance).
    """
    
    def __init__(
        self,
        input_size: int = 1000,
        latent_dim: int = 21,
        hidden_dims: list = [256, 128, 64]
    ):
        """
        Initialize VAE encoder.
        
        Parameters:
        -----------
        input_size : int
            Input sequence length
        latent_dim : int
            Dimension of latent space (21 factors as in FactorECG)
        hidden_dims : list
            Hidden layer dimensions
        """
        super(VAEEcgEncoder, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals (batch_size, seq_len, input_dim)
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (mu, logvar) - mean and log variance of latent distribution
        """
        # Flatten input if needed
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove feature dimension if present
        
        # Encode
        h = self.encoder(x)
        
        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEEcgDecoder(nn.Module):
    """
    Decoder network for VAE.
    Reconstructs ECG signals from latent representations.
    """
    
    def __init__(
        self,
        latent_dim: int = 21,
        output_size: int = 1000,
        hidden_dims: list = [64, 128, 256]
    ):
        """
        Initialize VAE decoder.
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of latent space
        output_size : int
            Output sequence length
        hidden_dims : list
            Hidden layer dimensions (reversed from encoder)
        """
        super(VAEEcgDecoder, self).__init__()
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_size))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent representation (batch_size, latent_dim)
        
        Returns:
        --------
        torch.Tensor
            Reconstructed ECG signals (batch_size, output_size)
        """
        return self.decoder(z)


class VAEEcgClassifier(nn.Module):
    """
    Variational Autoencoder for ECG Classification.
    
    This implementation follows the FactorECG approach, where the VAE
    learns a latent representation that can be used for both
    reconstruction and classification tasks.
    """
    
    def __init__(
        self,
        input_size: int = 1000,
        latent_dim: int = 21,
        num_classes: int = 5,
        hidden_dims: list = [256, 128, 64],
        beta: float = 0.001
    ):
        """
        Initialize VAE ECG Classifier.
        
        Parameters:
        -----------
        input_size : int
            Input sequence length
        latent_dim : int
            Dimension of latent space (21 factors as in FactorECG)
        num_classes : int
            Number of classification classes
        hidden_dims : list
            Hidden layer dimensions for encoder/decoder
        beta : float
            Beta parameter for beta-VAE (controls disentanglement)
        """
        super(VAEEcgClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = VAEEcgEncoder(
            input_size=input_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
        
        # Decoder
        self.decoder = VAEEcgDecoder(
            latent_dim=latent_dim,
            output_size=input_size,
            hidden_dims=list(reversed(hidden_dims))
        )
        
        # Classification head (uses latent representation)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        
        Returns:
        --------
        torch.Tensor
            Sampled latent representation
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through VAE.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input ECG signals (batch_size, seq_len, input_dim)
        return_reconstruction : bool
            Whether to return reconstructed signals
        
        Returns:
        --------
        Tuple containing:
        - Classification logits
        - (Optional) Reconstructed signals
        - (Optional) Latent mean
        - (Optional) Latent log variance
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode (for reconstruction loss)
        x_recon = self.decoder(z)
        
        # Classify (using latent representation)
        logits = self.classifier(mu)  # Use mean for classification
        
        if return_reconstruction:
            return logits, x_recon, mu, logvar
        else:
            return logits
    
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
        classification_loss: torch.Tensor
    ) -> dict:
        """
        Compute VAE loss (reconstruction + KL divergence + classification).
        
        Parameters:
        -----------
        x : torch.Tensor
            Original input signals
        x_recon : torch.Tensor
            Reconstructed signals
        mu : torch.Tensor
            Latent mean
        logvar : torch.Tensor
            Latent log variance
        labels : torch.Tensor
            True labels
        classification_loss : torch.Tensor
            Classification loss (cross-entropy)
        
        Returns:
        --------
        dict
            Dictionary containing individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x.squeeze(-1) if x.dim() == 3 else x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total loss (beta-VAE formulation)
        total_loss = recon_loss + self.beta * kl_loss + classification_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'classification_loss': classification_loss
        }


def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Train VAE model.
    
    Parameters:
    -----------
    model : nn.Module
        VAE model
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
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_class_loss': [],
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
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_class_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.squeeze().to(device)
        
            optimizer.zero_grad()
            
            # Forward pass
            logits, x_recon, mu, logvar = model(signals, return_reconstruction=True)
            
            # Classification loss
            class_loss = criterion(logits, labels)
            
            # VAE loss
            loss_dict = model.loss_function(signals, x_recon, mu, logvar, labels, class_loss)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_dict['recon_loss'].item()
            train_kl_loss += loss_dict['kl_loss'].item()
            train_class_loss += loss_dict['classification_loss'].item()
            
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_kl = train_kl_loss / len(train_loader)
        avg_train_class = train_class_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon)
        history['train_kl_loss'].append(avg_train_kl)
        history['train_class_loss'].append(avg_train_class)
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
                    
                    logits, x_recon, mu, logvar = model(signals, return_reconstruction=True)
                    class_loss = criterion(logits, labels)
                    loss_dict = model.loss_function(signals, x_recon, mu, logvar, labels, class_loss)
                    loss = loss_dict['total_loss']
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
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
    print("\nInitializing VAE ECG Classifier...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = VAEEcgClassifier(
        input_size=1000,
        latent_dim=21,  # 21 factors as in FactorECG
        num_classes=5,
        hidden_dims=[256, 128, 64],
        beta=0.001
    )
    
    # Train model
    print("\nTraining VAE model...")
    history = train_vae(
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

