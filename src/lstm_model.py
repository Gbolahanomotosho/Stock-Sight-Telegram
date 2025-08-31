# src/lstm_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

class SeqDataset(Dataset):
    """
    Dataset for time series sequences.
    X: shape (samples, sequence_length, features)
    y: shape (samples,)
    """
    def __init__(self, X, y):
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        # Convert and validate data types
        try:
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert data to float32: {e}")
        
        # Check for NaN values
        if np.isnan(self.X).any():
            print("[WARN] Found NaN values in X, replacing with zeros")
            self.X = np.nan_to_num(self.X, nan=0.0)
        
        if np.isnan(self.y).any():
            print("[WARN] Found NaN values in y, replacing with zeros")
            self.y = np.nan_to_num(self.y, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMNet(nn.Module):
    """
    Enhanced LSTM for regression on time series with better architecture.
    """
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Validate parameters
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if hidden <= 0:
            raise ValueError("hidden size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        self.hidden_size = hidden
        self.num_layers = num_layers
        
        # LSTM layer with improved configuration
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # No dropout for single layer
            bidirectional=False
        )
        
        # Enhanced output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)

    def forward(self, x):
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")
        
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last time step output
        last_output = out[:, -1, :]
        
        # Feed through dense layers
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)

def train_lstm(
    X_train, y_train,
    X_val, y_val,
    device="cpu",
    epochs=30,
    batch_size=64,
    lr=1e-3,
    patience=10
) -> Optional[LSTMNet]:
    """
    Train the LSTM model with early stopping and better error handling.
    """
    try:
        # Validate inputs
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training data provided")
        
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Empty validation data provided")
        
        print(f"[INFO] Training LSTM with {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Create datasets
        train_ds = SeqDataset(X_train, y_train)
        val_ds = SeqDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Ensure we have at least one batch
        if len(train_loader) == 0:
            raise ValueError("Training data too small for given batch size")
        
        # Initialize model
        model = LSTMNet(n_features=X_train.shape[-1]).to(device)
        
        # Optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for xb, yb in train_loader:
                try:
                    xb, yb = xb.to(device), yb.to(device)
                    
                    # Forward pass
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                except RuntimeError as e:
                    print(f"[WARN] Training batch failed: {e}")
                    continue
            
            if train_batches == 0:
                raise RuntimeError("All training batches failed")
            
            train_loss /= train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for xb, yb in val_loader:
                    try:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        val_loss += loss_fn(pred, yb).item()
                        val_batches += 1
                    except RuntimeError as e:
                        print(f"[WARN] Validation batch failed: {e}")
                        continue
            
            if val_batches > 0:
                val_loss /= val_batches
            else:
                val_loss = train_loss  # Fallback
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and best model tracking
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"[INFO] LSTM Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[INFO] LSTM training completed. Best validation loss: {best_loss:.6f}")
        else:
            print("[WARN] No improvement found during LSTM training, using final model")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] LSTM training failed: {e}")
        return None

def predict_lstm(model: Optional[LSTMNet], X, device="cpu") -> np.ndarray:
    """
    Predict using a trained LSTM model with error handling.
    """
    if model is None:
        print("[WARN] LSTM model is None, returning zeros")
        return np.zeros(len(X))
    
    try:
        if len(X) == 0:
            return np.array([])
        
        model.eval()
        
        # Convert to tensor with validation
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X.astype(np.float32))
        else:
            X_tensor = torch.tensor(np.array(X, dtype=np.float32))
        
        X_tensor = X_tensor.to(device)
        
        # Check for NaN values
        if torch.isnan(X_tensor).any():
            print("[WARN] Found NaN in LSTM input, replacing with zeros")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
        
        with torch.no_grad():
            predictions = model(X_tensor)
            
            # Validate predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("[WARN] LSTM predictions contain NaN/inf, replacing with zeros")
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return predictions.cpu().numpy()
            
    except Exception as e:
        print(f"[ERROR] LSTM prediction failed: {e}, returning zeros")
        return np.zeros(len(X))
