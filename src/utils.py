# src/utils.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union, List

def create_sliding_windows(values: np.ndarray, context: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from a time series for supervised learning.
    Args:
        values: shape (n_samples, n_features) or (n_samples,)
        context: number of past steps to include in each input window
        horizon: number of steps ahead to predict
    Returns:
        X: shape (n_windows, context, n_features)
        y: shape (n_windows, n_features) or (n_windows,) if single target
    """
    # Validate inputs
    if values is None or values.size == 0:
        raise ValueError("Values array cannot be empty")
    
    if context <= 0:
        raise ValueError("Context must be positive")
    
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    
    # Ensure 2D array
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim != 2:
        raise ValueError(f"Values must be 1D or 2D, got {values.ndim}D")
    
    n_samples, n_features = values.shape
    
    # Check if we have enough data
    min_required = context + horizon
    if n_samples < min_required:
        raise ValueError(f"Not enough data: need at least {min_required} samples, got {n_samples}")
    
    # Calculate number of windows
    n_windows = n_samples - context - horizon + 1
    
    if n_windows <= 0:
        raise ValueError(f"Cannot create windows: context={context}, horizon={horizon}, samples={n_samples}")
    
    X = []
    y = []
    
    for i in range(n_windows):
        # Input window: past context steps
        window_start = i
        window_end = i + context
        X.append(values[window_start:window_end])
        
        # Target: value at horizon steps ahead
        target_idx = i + context + horizon - 1
        y.append(values[target_idx])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Validate output shapes
    expected_X_shape = (n_windows, context, n_features)
    if X.shape != expected_X_shape:
        raise ValueError(f"Unexpected X shape: expected {expected_X_shape}, got {X.shape}")
    
    # Handle y shape based on number of features
    if n_features == 1:
        y = y.squeeze(-1)  # Remove last dimension if single feature
    
    return X, y

def scale_train_val_test(X_train: np.ndarray, X_val: np.ndarray, X_test: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray, List[StandardScaler]], Tuple[np.ndarray, np.ndarray, np.ndarray, List[StandardScaler]]]:
    """
    Scale features per feature dimension using StandardScaler.
    Args:
        X_train: (n_train, seq_len, n_features)
        X_val: (n_val, seq_len, n_features)
        X_test: optional (n_test, seq_len, n_features)
    Returns:
        X_train_scaled, X_val_scaled, (X_test_scaled if provided), scalers
    """
    # Validate inputs
    if X_train is None or X_train.size == 0:
        raise ValueError("X_train cannot be empty")
    
    if X_val is None or X_val.size == 0:
        raise ValueError("X_val cannot be empty")
    
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D, got {X_train.ndim}D")
    
    if X_val.ndim != 3:
        raise ValueError(f"X_val must be 3D, got {X_val.ndim}D")
    
    if X_test is not None and X_test.ndim != 3:
        raise ValueError(f"X_test must be 3D, got {X_test.ndim}D")
    
    # Check feature dimension consistency
    n_features = X_train.shape[-1]
    if X_val.shape[-1] != n_features:
        raise ValueError(f"Feature dimension mismatch: X_train={n_features}, X_val={X_val.shape[-1]}")
    
    if X_test is not None and X_test.shape[-1] != n_features:
        raise ValueError(f"Feature dimension mismatch: X_train={n_features}, X_test={X_test.shape[-1]}")
    
    scalers = []
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    if X_test is not None:
        X_test_scaled = X_test.copy()
    
    # Scale each feature separately
    for f in range(n_features):
        try:
            scaler = StandardScaler()
            
            # Fit scaler on training data for this feature
            # Reshape from (n_samples, seq_len) to (n_samples * seq_len,) for fitting
            train_feature_data = X_train[:, :, f].reshape(-1, 1)
            
            # Remove any NaN or infinite values for fitting
            finite_mask = np.isfinite(train_feature_data.flatten())
            if not finite_mask.any():
                print(f"[WARN] All values are NaN/inf for feature {f}, using identity scaling")
                # Create identity scaler
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
            else:
                finite_data = train_feature_data[finite_mask.reshape(-1, 1)]
                scaler.fit(finite_data)
            
            # Transform training data
            X_train_scaled[:, :, f] = scaler.transform(X_train[:, :, f].reshape(-1, 1)).reshape(X_train[:, :, f].shape)
            
            # Transform validation data
            X_val_scaled[:, :, f] = scaler.transform(X_val[:, :, f].reshape(-1, 1)).reshape(X_val[:, :, f].shape)
            
            # Transform test data if provided
            if X_test is not None:
                X_test_scaled[:, :, f] = scaler.transform(X_test[:, :, f].reshape(-1, 1)).reshape(X_test[:, :, f].shape)
            
            scalers.append(scaler)
            
        except Exception as e:
            print(f"[WARN] Scaling failed for feature {f}: {e}, using identity scaling")
            # Create identity scaler as fallback
            identity_scaler = StandardScaler()
            identity_scaler.mean_ = np.array([0.0])
            identity_scaler.scale_ = np.array([1.0])
            scalers.append(identity_scaler)
    
    # Clean scaled data
    for arr in [X_train_scaled, X_val_scaled] + ([X_test_scaled] if X_test is not None else []):
        # Replace any remaining NaN/inf with reasonable values
        if np.isnan(arr).any() or np.isinf(arr).any():
            print("[WARN] Found NaN/inf in scaled data, cleaning...")
            arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
    
    if X_test is not None:
        return X_train_scaled, X_val_scaled, X_test_scaled, scalers
    else:
        return X_train_scaled, X_val_scaled, scalers

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error with input validation.
    """
    try:
        # Validate inputs
        if actual is None or predicted is None:
            raise ValueError("Actual and predicted arrays cannot be None")
        
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        if actual.shape != predicted.shape:
            raise ValueError(f"Shape mismatch: actual={actual.shape}, predicted={predicted.shape}")
        
        if actual.size == 0:
            return 0.0
        
        # Clean data
        finite_mask = np.isfinite(actual) & np.isfinite(predicted)
        if not finite_mask.any():
            print("[WARN] No finite values in RMSE calculation")
            return float('inf')
        
        if finite_mask.sum() < len(actual):
            print(f"[WARN] {len(actual) - finite_mask.sum()} non-finite values removed from RMSE calculation")
            actual = actual[finite_mask]
            predicted = predicted[finite_mask]
        
        # Calculate RMSE
        mse = np.mean((actual - predicted) ** 2)
        return float(np.sqrt(mse))
        
    except Exception as e:
        print(f"[ERROR] RMSE calculation failed: {e}")
        return float('inf')

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error with enhanced input validation and handling of zero values.
    """
    try:
        # Validate inputs
        if actual is None or predicted is None:
            raise ValueError("Actual and predicted arrays cannot be None")
        
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        if actual.shape != predicted.shape:
            raise ValueError(f"Shape mismatch: actual={actual.shape}, predicted={predicted.shape}")
        
        if actual.size == 0:
            return 0.0
        
        # Clean data
        finite_mask = np.isfinite(actual) & np.isfinite(predicted)
        if not finite_mask.any():
            print("[WARN] No finite values in MAPE calculation")
            return float('inf')
        
        if finite_mask.sum() < len(actual):
            print(f"[WARN] {len(actual) - finite_mask.sum()} non-finite values removed from MAPE calculation")
            actual = actual[finite_mask]
            predicted = predicted[finite_mask]
        
        # Handle zero values in actual (common issue with MAPE)
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-8
        denominator = np.where(np.abs(actual) < epsilon, epsilon, actual)
        
        # Calculate MAPE
        percentage_errors = np.abs((actual - predicted) / denominator)
        
        # Remove any remaining infinite values
        finite_errors = percentage_errors[np.isfinite(percentage_errors)]
        
        if len(finite_errors) == 0:
            print("[WARN] All percentage errors are non-finite in MAPE calculation")
            return float('inf')
        
        return float(np.mean(finite_errors) * 100.0)
        
    except Exception as e:
        print(f"[ERROR] MAPE calculation failed: {e}")
        return float('inf')

def validate_array(arr: np.ndarray, name: str, expected_shape: Optional[Tuple] = None, allow_empty: bool = False) -> bool:
    """
    Validate numpy array with comprehensive checks.
    """
    try:
        if arr is None:
            raise ValueError(f"{name} cannot be None")
        
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        
        if not allow_empty and arr.size == 0:
            raise ValueError(f"{name} cannot be empty")
        
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {arr.shape}")
        
        # Check for problematic values
        if np.isnan(arr).all():
            raise ValueError(f"All values in {name} are NaN")
        
        if np.isinf(arr).any():
            print(f"[WARN] {name} contains infinite values")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Validation failed for {name}: {e}")
        return False

def clean_predictions(predictions: np.ndarray, method: str = "clip") -> np.ndarray:
    """
    Clean prediction arrays by handling NaN, inf, and extreme values.
    """
    try:
        if predictions is None or predictions.size == 0:
            return predictions
        
        predictions = np.asarray(predictions, dtype=np.float64)
        
        # Handle NaN and infinite values
        if method == "clip":
            # Clip extreme values to reasonable range
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        elif method == "remove":
            # Remove NaN/inf values (changes array size)
            finite_mask = np.isfinite(predictions)
            predictions = predictions[finite_mask]
        elif method == "interpolate":
            # Simple forward/backward fill for NaN values
            if np.isnan(predictions).any():
                # Forward fill
                for i in range(1, len(predictions)):
                    if np.isnan(predictions[i]) and not np.isnan(predictions[i-1]):
                        predictions[i] = predictions[i-1]
                
                # Backward fill for remaining NaN at the beginning
                for i in range(len(predictions)-2, -1, -1):
                    if np.isnan(predictions[i]) and not np.isnan(predictions[i+1]):
                        predictions[i] = predictions[i+1]
                
                # If still NaN remain, replace with zeros
                predictions = np.nan_to_num(predictions, nan=0.0)
        
        return predictions.astype(np.float32)
        
    except Exception as e:
        print(f"[ERROR] Prediction cleaning failed: {e}")
        return np.zeros_like(predictions) if predictions is not None else np.array([])

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default_value: float = 0.0) -> np.ndarray:
    """
    Safely divide two arrays, handling division by zero and NaN values.
    """
    try:
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
        
        # Replace inf and NaN with default value
        result = np.where(np.isfinite(result), result, default_value)
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Safe division failed: {e}")
        return np.full_like(numerator, default_value)
