# src/timesnet_model.py
# Enhanced TimesNet-style block in PyTorch (period-aware temporal convolutions)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Tuple

class PeriodBlock(nn.Module):
    def __init__(self, n_features, d_model=64, periods=(7, 14, 30)):
        super().__init__()
        
        # Validate parameters
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not periods:
            raise ValueError("periods cannot be empty")
        
        self.in_proj = nn.Linear(n_features, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        # Create conv layers for different periods
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=max(3, min(p // 2, 7)), padding='same')
            for p in periods
        ])
        
        # Batch normalization for each conv
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(d_model) for _ in periods
        ])
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # Output projection with residual connection
        self.layer_norm2 = nn.LayerNorm(d_model * len(periods))
        self.out_proj = nn.Linear(d_model * len(periods), 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # x: (B, T, F)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, time, features), got {x.dim()}D")
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection and normalization
        z = self.in_proj(x)  # (B, T, d_model)
        z = self.layer_norm1(z)
        z = self.act(z)
        
        # Transpose for conv1d: (B, T, d_model) -> (B, d_model, T)
        z = z.transpose(1, 2)
        
        # Apply convolutions for different periods
        conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            try:
                conv_out = conv(z)  # (B, d_model, T)
                
                # Apply batch norm only if batch size > 1
                if batch_size > 1:
                    conv_out = bn(conv_out)
                
                conv_out = self.act(conv_out)
                conv_outputs.append(conv_out)
            except RuntimeError as e:
                print(f"[WARN] Conv layer {i} failed: {e}, using zeros")
                conv_outputs.append(torch.zeros_like(z))
        
        # Concatenate all period outputs
        if conv_outputs:
            h = torch.cat(conv_outputs, dim=1)  # (B, d_model*P, T)
        else:
            # Fallback if all convs failed
            h = z.repeat(1, len(self.convs), 1)
        
        # Transpose back: (B, d_model*P, T) -> (B, T, d_model*P)
        h = h.transpose(1, 2)
        
        # Apply layer norm and dropout
        h = self.layer_norm2(h)
        h = self.dropout(h)
        
        # Use last time step for regression
        last_step = h[:, -1, :]  # (B, d_model*P)
        
        # Final output projection
        output = self.out_proj(last_step)  # (B, 1)
        return output.squeeze(-1)  # (B,)

class TimesNet(nn.Module):
    def __init__(self, n_features, d_model=64, periods=(7, 14, 30), num_blocks=1):
        super().__init__()
        
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        
        # Stack multiple period blocks for better representation
        self.blocks = nn.ModuleList([
            PeriodBlock(n_features if i == 0 else 1, d_model, periods) 
            for i in range(num_blocks)
        ])
        
        self.num_blocks = num_blocks

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, time, features), got {x.dim()}D")
        
        # Process through period blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                output = block(x)  # (B,)
            else:
                # For subsequent blocks, use previous output as input
                # Expand dims to match expected input shape
                block_input = output.unsqueeze(-1).unsqueeze(-1).expand(-1, x.size(1), -1)
                output = block(block_input)
        
        return output

def train_timesnet(
    X_train, y_train, 
    X_val, y_val, 
    device="cpu", 
    epochs=30, 
    batch_size=64, 
    lr=1e-3,
    patience=10
) -> Optional[TimesNet]:
    """
    Train TimesNet model with enhanced error handling and early stopping.
    """
    try:
        # Validate inputs
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training data provided")
        
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Empty validation data provided")
        
        print(f"[INFO] Training TimesNet with {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Initialize model
        model = TimesNet(n_features=X_train.shape[-1]).to(device)
        
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
                    
                    # Clean inputs
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
                print(f"[INFO] TimesNet Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[INFO] TimesNet training completed. Best validation loss: {best_loss:.6f}")
        else:
            print("[WARN] No improvement found during TimesNet training, using final model")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] TimesNet training failed: {e}")
        return None

def predict_timesnet(model: Optional[TimesNet], X, device="cpu") -> np.ndarray:
    """
    Predict using a trained TimesNet model with error handling.
    """
    if model is None:
        print("[WARN] TimesNet model is None, returning zeros")
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
            print("[WARN] Found NaN/inf in TimesNet input, cleaning...")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        with torch.no_grad():
            predictions = model(X_tensor)
            
            # Validate predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("[WARN] TimesNet predictions contain NaN/inf, cleaning...")
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return predictions.cpu().numpy()
            
    except Exception as e:
        print(f"[ERROR] TimesNet prediction failed: {e}, returning zeros")
        return np.zeros(len(X))
