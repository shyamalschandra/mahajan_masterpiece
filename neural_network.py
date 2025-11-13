"""
Neural Network Implementation for ECG/Heart Disease Prediction
Based on: Detection of Ischemia in the Electrocardiogram Using Artificial Neural Networks
(Michael D. Lloyd et al., Circulation, 2001)

This implementation provides a feedforward neural network with backpropagation
suitable for binary classification tasks (e.g., ischemia detection, mortality prediction).
"""

import numpy as np                                                      # NumPy for array operations
import matplotlib.pyplot as plt                                        # Matplotlib for plotting
from typing import List, Tuple, Optional                                # Type hints


class NeuralNetwork:
    """
    Feedforward Neural Network with configurable architecture.
    Supports multiple hidden layers with various activation functions.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str = 'sigmoid',
        learning_rate: float = 0.01,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_layers : List[int]
            List specifying the number of neurons in each hidden layer
            e.g., [16, 8] means two hidden layers with 16 and 8 neurons
        output_size : int
            Number of output classes (typically 1 for binary classification)
        activation : str
            Activation function to use ('sigmoid', 'tanh', 'relu')
        learning_rate : float
            Learning rate for gradient descent
        random_seed : int, optional
            Random seed for reproducibility
        """
        if random_seed is not None:                                      # Set random seed if provided
            np.random.seed(random_seed)                                  # For reproducibility
        
        self.input_size = input_size                                     # Store input dimension
        self.hidden_layers = hidden_layers                               # Store hidden layer sizes
        self.output_size = output_size                                   # Store output dimension
        self.activation = activation                                     # Store activation function type
        self.learning_rate = learning_rate                               # Store learning rate
        
        # Build network architecture
        self.architecture = [input_size] + hidden_layers + [output_size] # Complete architecture list
        self.num_layers = len(self.architecture) - 1                    # Total number of weight layers
        
        # Initialize weights and biases
        self.weights = []                                                # List to store weight matrices
        self.biases = []                                                 # List to store bias vectors
        
        for i in range(self.num_layers):
            # Xavier/Glorot initialization for better convergence
            weight_matrix = np.random.randn(                            # Initialize weights randomly
                self.architecture[i],
                self.architecture[i + 1]
            ) * np.sqrt(2.0 / self.architecture[i])                      # Scale by sqrt(2/n) for better convergence
            
            bias_vector = np.zeros((1, self.architecture[i + 1]))        # Initialize biases to zero
            
            self.weights.append(weight_matrix)                          # Store weight matrix
            self.biases.append(bias_vector)                             # Store bias vector
        
        # Storage for forward propagation values
        self.activations = []                                            # Store activation values for backprop
        self.z_values = []                                               # Store pre-activation values
        
        # Training history
        self.loss_history = []                                           # Track loss over epochs
        self.accuracy_history = []                                       # Track accuracy over epochs
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'sigmoid':
            # Clip values to prevent overflow
            x = np.clip(x, -500, 500)                                    # Prevent numerical overflow
            return 1 / (1 + np.exp(-x))                                  # Sigmoid: 1/(1+e^-x)
        elif self.activation == 'tanh':
            return np.tanh(x)                                            # Hyperbolic tangent
        elif self.activation == 'relu':
            return np.maximum(0, x)                                      # ReLU: max(0, x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'sigmoid':
            s = self._activation(x)                                      # Get activation value
            return s * (1 - s)                                          # Derivative: s(1-s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2                                  # Derivative: 1-tanh²(x)
        elif self.activation == 'relu':
            return (x > 0).astype(float)                                # Derivative: 1 if x>0, else 0
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def _sigmoid_output(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid for output layer (binary classification)."""
        x = np.clip(x, -500, 500)                                       # Clip to prevent overflow
        return 1 / (1 + np.exp(-x))                                     # Sigmoid for binary output
    
    def _sigmoid_output_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid for output layer."""
        s = self._sigmoid_output(x)                                      # Get sigmoid value
        return s * (1 - s)                                              # Return derivative
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Network output predictions
        """
        # Reset storage
        self.activations = [X]                                          # Store input as first activation
        self.z_values = []                                               # Clear pre-activation values
        
        current_input = X                                               # Start with input data
        
        # Forward pass through hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i] # Compute weighted sum: Wx + b
            self.z_values.append(z)                                      # Store for backprop
            a = self._activation(z)                                      # Apply activation function
            self.activations.append(a)                                   # Store activation
            current_input = a                                            # Use as input for next layer
        
        # Output layer (always uses sigmoid for binary classification)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1] # Final weighted sum
        self.z_values.append(z_output)                                   # Store output pre-activation
        output = self._sigmoid_output(z_output)                         # Apply sigmoid for probability
        self.activations.append(output)                                  # Store final output
        
        return output                                                    # Return predictions
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Perform backward propagation to compute gradients and update weights.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True labels
        output : np.ndarray
            Network predictions from forward propagation
        """
        m = X.shape[0]                                                  # Number of samples in batch
        
        # Compute output layer error
        delta = output - y                                              # Error signal: prediction - target
        
        # Backpropagate through layers
        for i in range(self.num_layers - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m               # Weight gradient: (1/m) * a^T * delta
            db = np.sum(delta, axis=0, keepdims=True) / m               # Bias gradient: (1/m) * sum(delta)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW                   # Update weights: W = W - α*dW
            self.biases[i] -= self.learning_rate * db                    # Update biases: b = b - α*db
            
            # Backpropagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)                # Propagate error: delta * W^T
                delta *= self._activation_derivative(self.z_values[i - 1]) # Multiply by activation derivative
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        
        Returns:
        --------
        float
            Loss value
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)                     # Prevent log(0) errors
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # Binary cross-entropy
        return loss                                                     # Return loss value
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        threshold : float
            Classification threshold
        
        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1)
        """
        probabilities = self.forward_propagation(X)                     # Get probability predictions
        return (probabilities >= threshold).astype(int)                 # Convert to binary: 0 or 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray
            Probability predictions
        """
        return self.forward_propagation(X)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        
        Returns:
        --------
        float
            Accuracy score
        """
        return np.mean(y_true == y_pred)                                 # Calculate accuracy: mean of matches
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 10
    ) -> dict:
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation input data
        y_val : np.ndarray, optional
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int, optional
            Batch size for mini-batch gradient descent (None = full batch)
        verbose : bool
            Whether to print training progress
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Early stopping patience (epochs without improvement)
        
        Returns:
        --------
        dict
            Training history
        """
        if batch_size is None:
            batch_size = X_train.shape[0]                               # Use full batch if not specified
        
        best_val_loss = float('inf')                                     # Track best validation loss
        patience_counter = 0                                              # Counter for early stopping
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(X_train.shape[0])            # Shuffle training data
            X_shuffled = X_train[indices]                                # Shuffled features
            y_shuffled = y_train[indices]                                # Shuffled labels
            
            epoch_loss = 0                                               # Accumulate loss this epoch
            num_batches = 0                                              # Count batches
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]                   # Extract batch of features
                y_batch = y_shuffled[i:i + batch_size]                   # Extract batch of labels
                
                # Forward propagation
                output = self.forward_propagation(X_batch)                # Get predictions
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, output)          # Calculate batch loss
                epoch_loss += batch_loss                                 # Add to epoch total
                num_batches += 1                                         # Increment batch counter
                
                # Backward propagation
                self.backward_propagation(X_batch, y_batch, output)     # Update weights via backprop
            
            avg_loss = epoch_loss / num_batches                           # Average loss for epoch
            
            # Evaluate on training set
            train_pred = self.predict(X_train)                          # Predict on training data
            train_acc = self.compute_accuracy(y_train, train_pred)       # Calculate training accuracy
            
            self.loss_history.append(avg_loss)                           # Store loss
            self.accuracy_history.append(train_acc)                       # Store accuracy
            
            # Validation evaluation
            val_loss = None                                              # Initialize validation loss
            val_acc = None                                               # Initialize validation accuracy
            if X_val is not None and y_val is not None:
                val_output = self.forward_propagation(X_val)            # Predict on validation data
                val_loss = self.compute_loss(y_val, val_output)          # Calculate validation loss
                val_pred = self.predict(X_val)                           # Get validation predictions
                val_acc = self.compute_accuracy(y_val, val_pred)        # Calculate validation accuracy
            
            # Early stopping
            if early_stopping and val_loss is not None:
                if val_loss < best_val_loss:                             # Check if validation improved
                    best_val_loss = val_loss                             # Update best loss
                    patience_counter = 0                                 # Reset patience counter
                else:
                    patience_counter += 1                                # Increment patience counter
                    if patience_counter >= patience:                     # Check if patience exceeded
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                        break                                            # Stop training
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:                       # Print every 100 epochs
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                print(msg)                                               # Print training progress
        
        return {
            'loss_history': self.loss_history,                           # Return loss over time
            'accuracy_history': self.accuracy_history                    # Return accuracy over time
        }
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training loss and accuracy history."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)                 # Create 2 subplots side by side
        
        axes[0].plot(self.loss_history)                                  # Plot loss curve
        axes[0].set_title('Training Loss')                              # Set title
        axes[0].set_xlabel('Epoch')                                      # X-axis label
        axes[0].set_ylabel('Loss')                                       # Y-axis label
        axes[0].grid(True)                                               # Add grid
        
        axes[1].plot(self.accuracy_history)                              # Plot accuracy curve
        axes[1].set_title('Training Accuracy')                           # Set title
        axes[1].set_xlabel('Epoch')                                      # X-axis label
        axes[1].set_ylabel('Accuracy')                                   # Y-axis label
        axes[1].grid(True)                                               # Add grid
        
        plt.tight_layout()                                               # Adjust spacing
        plt.show()                                                       # Display plot


def create_sample_ecg_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample ECG/heart disease dataset for demonstration.
    In practice, replace this with real ECG data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (e.g., ECG features, patient demographics)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (features) and y (labels)
    """
    np.random.seed(42)                                                  # Set seed for reproducibility
    
    # Generate synthetic features (replace with real ECG features)
    X = np.random.randn(n_samples, n_features)                           # Random Gaussian features
    
    # Create labels with some relationship to features
    # In practice, labels come from medical diagnosis
    y = (np.sum(X[:, :3], axis=1) > 0.5).astype(int).reshape(-1, 1)    # Binary labels based on first 3 features
    
    return X, y                                                          # Return features and labels


if __name__ == "__main__":
    # Example usage
    print("Creating sample ECG dataset...")
    X, y = create_sample_ecg_data(n_samples=1000, n_features=10)       # Generate synthetic data
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(X))                                      # 80% train, 20% validation
    X_train, X_val = X[:split_idx], X[split_idx:]                      # Split features
    y_train, y_val = y[:split_idx], y[split_idx:]                      # Split labels
    
    # Normalize features
    mean = X_train.mean(axis=0)                                        # Compute mean per feature
    std = X_train.std(axis=0)                                          # Compute std per feature
    X_train = (X_train - mean) / std                                   # Normalize training data
    X_val = (X_val - mean) / std                                       # Normalize validation data
    
    print("\nInitializing Neural Network...")
    print("Architecture: Input(10) -> Hidden(16) -> Hidden(8) -> Output(1)")
    
    # Initialize network
    nn = NeuralNetwork(
        input_size=10,                                                  # 10 input features
        hidden_layers=[16, 8],                                          # 2 hidden layers: 16 and 8 neurons
        output_size=1,                                                  # Binary classification output
        activation='sigmoid',                                            # Sigmoid activation
        learning_rate=0.01,                                              # Learning rate
        random_seed=42                                                   # Random seed
    )
    
    # Train network
    print("\nTraining Neural Network...")
    history = nn.train(
        X_train, y_train,                                                # Training data
        X_val, y_val,                                                   # Validation data
        epochs=1000,                                                    # Train for 1000 epochs
        batch_size=32,                                                  # Mini-batch size
        verbose=True,                                                   # Print progress
        early_stopping=True,                                            # Enable early stopping
        patience=20                                                     # Wait 20 epochs without improvement
    )
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_predictions = nn.predict(X_val)                                 # Get binary predictions
    val_accuracy = nn.compute_accuracy(y_val, val_predictions)          # Calculate accuracy
    val_proba = nn.predict_proba(X_val)                                # Get probability predictions
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")                  # Print accuracy
    print(f"Sample predictions (first 10): {val_predictions.flatten()[:10]}") # Show sample predictions
    print(f"Sample probabilities (first 10): {val_proba.flatten()[:10]}") # Show sample probabilities
    
    # Plot training history
    print("\nPlotting training history...")
    nn.plot_training_history()                                          # Display training curves

