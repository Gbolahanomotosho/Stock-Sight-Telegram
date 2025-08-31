# src/transformer_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Add positional encoding to improve transformer performance on sequential data"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeTransformer(nn.Module):
    """
    Enhanced Transformer encoder for time series regression.
    """
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # Validate parameters
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # (seq, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")
        
        # x: (batch, seq, features)
        x = self.input_proj(x) * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        
        # Convert to (seq, batch, d_model) for transformer
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use the last time step output: (seq, batch, d_model) -> (batch, d_model)
        last_output = encoded[-1]
        
        # Feed through dense layers
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output.squeeze(-1)

def train_transformer(
    X_train, y_train,
    X_val, y_val,
    device="cpu",
    epochs=30,
    batch_size=64,
    lr=1e-3,
    patience=10
) -> Optional[TimeTransformer]:
    """
    Train a Transformer model for time series forecasting with enhanced error handling.
    """
    try:
        # Validate inputs
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training data provided")
        
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Empty validation data provided")
        
        print(f"[INFO] Training Transformer with {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Initialize model
        model = TimeTransformer(n_features=X_train.shape[-1]).to(device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Create datasets with validation
        try:
            train_ds = TensorDataset(
                torch.tensor(X_train.astype(np.float32)),
                torch.tensor(y_train.astype(np.float32))
            )
            val_ds = TensorDataset(
                torch.tensor(X_val.astype(np.float32)),
                torch.tensor(y_val.astype(np.float32))
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert data to tensors: {e}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Ensure we have at least one batch
        if len(train_loader) == 0:
            raise ValueError("Training data too small for given batch size")
        
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
                    
                    # Check for NaN/inf in inputs
                    if torch.isnan(xb).any() or torch.isinf(xb).any():
                        xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
                    if torch.isnan(yb).any() or torch.isinf(yb).any():
                        yb = torch.nan_to_num(yb, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    # Forward pass
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARN] NaN/inf loss detected, skipping batch")
                        continue
                    
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
                        
                        # Clean inputs
                        if torch.isnan(xb).any() or torch.isinf(xb).any():
                            xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
                        if torch.isnan(yb).any() or torch.isinf(yb).any():
                            yb = torch.nan_to_num(yb, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            val_loss += loss.item()
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
                print(f"[INFO] Transformer Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[INFO] Transformer training completed. Best validation loss: {best_loss:.6f}")
        else:
            print("[WARN] No improvement found during Transformer training, using final model")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Transformer training failed: {e}")
        return None

def predict_transformer(model: Optional[TimeTransformer], X, device="cpu") -> np.ndarray:
    """
    Predict using a trained Transformer model with error handling.
    """
    if model is None:
        print("[WARN] Transformer model is None, returning zeros")
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
        if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
            print("[WARN] Found NaN/inf in Transformer input, cleaning...")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        with torch.no_grad():
            predictions = model(X_tensor)
            
            # Validate predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("[WARN] Transformer predictions contain NaN/inf, cleaning...")
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return predictions.cpu().numpy()
            
    except Exception as e:
        print(f"[ERROR] Transformer prediction failed: {e}, returning zeros")
        return np.zeros(len(X))
