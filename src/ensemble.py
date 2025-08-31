# src/ensemble.py
import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional

def fit_meta(preds_matrix: np.ndarray, y_true: np.ndarray, alpha: float = 1.0) -> Optional[Ridge]:
    """
    Fit a Ridge regression meta-learner on predictions from base models.
    Args:
        preds_matrix: shape (n_samples, n_models) with predictions from base models.
        y_true: shape (n_samples,) actual target values.
        alpha: Ridge regularization strength.
    Returns:
        Trained Ridge model or None if fitting fails.
    """
    try:
        # Validate inputs
        if preds_matrix.size == 0 or y_true.size == 0:
            raise ValueError("Empty prediction matrix or target array")
        
        if len(preds_matrix) != len(y_true):
            raise ValueError(f"Length mismatch: preds_matrix={len(preds_matrix)}, y_true={len(y_true)}")
        
        if preds_matrix.ndim != 2:
            raise ValueError(f"preds_matrix must be 2D, got shape {preds_matrix.shape}")
        
        # Check for NaN or infinite values
        if np.isnan(preds_matrix).any() or np.isinf(preds_matrix).any():
            print("[WARN] Found NaN/inf in prediction matrix, cleaning...")
            preds_matrix = np.nan_to_num(preds_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            print("[WARN] Found NaN/inf in target values, cleaning...")
            y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure minimum samples for ridge regression
        if len(y_true) < preds_matrix.shape[1]:
            print(f"[WARN] Few samples ({len(y_true)}) for {preds_matrix.shape[1]} features, using higher alpha")
            alpha = max(alpha, 10.0)
        
        meta = Ridge(alpha=alpha, fit_intercept=True)
        meta.fit(preds_matrix, y_true)
        
        # Validate the fitted model
        if hasattr(meta, 'coef_') and np.isnan(meta.coef_).any():
            raise ValueError("Meta-learner coefficients contain NaN values")
        
        return meta
        
    except Exception as e:
        print(f"[ERROR] Meta-learner fitting failed: {e}")
        return None

def predict_meta(meta: Ridge, preds_matrix: np.ndarray) -> np.ndarray:
    """
    Predict using a trained Ridge meta-learner.
    Args:
        meta: trained Ridge model.
        preds_matrix: shape (n_samples, n_models) with predictions from base models.
    Returns:
        np.ndarray of final blended predictions, or simple average if meta prediction fails.
    """
    try:
        # Validate inputs
        if meta is None:
            print("[WARN] Meta-learner is None, using simple average")
            return np.mean(preds_matrix, axis=1)
        
        if preds_matrix.size == 0:
            return np.array([])
        
        if preds_matrix.ndim != 2:
            raise ValueError(f"preds_matrix must be 2D, got shape {preds_matrix.shape}")
        
        # Clean input data
        if np.isnan(preds_matrix).any() or np.isinf(preds_matrix).any():
            print("[WARN] Found NaN/inf in prediction matrix for meta prediction, cleaning...")
            preds_matrix = np.nan_to_num(preds_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Make predictions
        predictions = meta.predict(preds_matrix)
        
        # Validate output
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("[WARN] Meta predictions contain NaN/inf, using simple average fallback")
            return np.mean(preds_matrix, axis=1)
        
        return predictions
        
    except Exception as e:
        print(f"[ERROR] Meta prediction failed: {e}, using simple average fallback")
        return np.mean(preds_matrix, axis=1)
