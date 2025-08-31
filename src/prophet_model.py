# src/prophet_model.py
from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Optional
import warnings

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")

def train_prophet(
    df: pd.DataFrame,
    country_code: Optional[str] = None,
    changepoint_prior_scale: float = 0.05,
    daily_seasonality: bool = False,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True
) -> Optional[Prophet]:
    """
    Train a Prophet model with enhanced error handling and validation.
    """
    try:
        # Validate input dataframe
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty")
        
        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and validate data
        df_clean = df[['ds', 'y']].copy()
        
        # Ensure ds is datetime
        df_clean['ds'] = pd.to_datetime(df_clean['ds'])
        
        # Ensure y is numeric and finite
        df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')
        
        # Remove rows with NaN values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after cleaning")
        
        if len(df_clean) < initial_rows * 0.5:
            print(f"[WARN] Removed {initial_rows - len(df_clean)} rows with missing values")
        
        # Ensure minimum data points for Prophet
        if len(df_clean) < 10:
            raise ValueError(f"Insufficient data for Prophet: {len(df_clean)} points (need at least 10)")
        
        # Check for duplicate dates
        duplicate_dates = df_clean['ds'].duplicated().sum()
        if duplicate_dates > 0:
            print(f"[WARN] Found {duplicate_dates} duplicate dates, keeping last values")
            df_clean = df_clean.drop_duplicates(subset=['ds'], keep='last')
        
        # Sort by date
        df_clean = df_clean.sort_values('ds').reset_index(drop=True)
        
        # Check for constant values (Prophet fails with constant data)
        if df_clean['y'].std() == 0:
            print("[WARN] Constant values detected, adding small noise for Prophet")
            df_clean['y'] += np.random.normal(0, df_clean['y'].mean() * 0.001, len(df_clean))
        
        # Initialize Prophet model with conservative parameters
        m = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode='multiplicative' if df_clean['y'].min() > 0 else 'additive',
            interval_width=0.80,  # Slightly narrower intervals for stability
            mcmc_samples=0,  # Disable MCMC for faster training
            uncertainty_samples=1000
        )
        
        # Add country holidays if specified
        if country_code:
            try:
                m.add_country_holidays(country_name=country_code)
                print(f"[INFO] Added holidays for {country_code}")
            except Exception as e:
                print(f"[WARN] Failed to add holidays for {country_code}: {e}")
        
        # Fit the model with error handling
        print(f"[INFO] Training Prophet on {len(df_clean)} data points...")
        
        # Suppress Prophet's verbose output
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        m.fit(df_clean, verbose=False)
        
        # Validate the fitted model
        if not hasattr(m, 'params') or len(m.params) == 0:
            raise RuntimeError("Prophet model training failed - no parameters learned")
        
        print("[INFO] Prophet training completed successfully")
        return m
        
    except Exception as e:
        print(f"[ERROR] Prophet training failed: {e}")
        return None

def prophet_predict(
    m: Optional[Prophet],
    periods: int = 1,
    freq: str = "D",
    include_history: bool = True
) -> Optional[pd.DataFrame]:
    """
    Generate predictions from a trained Prophet model with error handling.
    """
    try:
        if m is None:
            print("[ERROR] Prophet model is None")
            return None
        
        if periods <= 0:
            raise ValueError("Periods must be positive")
        
        if periods > 365:
            print(f"[WARN] Large prediction horizon ({periods} periods), limiting to 365")
            periods = 365
        
        # Generate future dataframe
        future = m.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        
        if future.empty:
            raise ValueError("Failed to create future dataframe")
        
        # Make predictions with error handling
        print(f"[INFO] Generating Prophet predictions for {periods} periods...")
        
        forecast = m.predict(future)
        
        if forecast.empty:
            raise ValueError("Prophet prediction returned empty results")
        
        # Validate predictions
        if 'yhat' not in forecast.columns:
            raise ValueError("Prophet forecast missing 'yhat' column")
        
        # Check for invalid predictions
        invalid_preds = forecast['yhat'].isna().sum()
        if invalid_preds > 0:
            print(f"[WARN] Found {invalid_preds} NaN predictions, filling with last valid value")
            forecast['yhat'] = forecast['yhat'].fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        inf_preds = np.isinf(forecast['yhat']).sum()
        if inf_preds > 0:
            print(f"[WARN] Found {inf_preds} infinite predictions, clipping values")
            forecast['yhat'] = forecast['yhat'].replace([np.inf, -np.inf], np.nan)
            forecast['yhat'] = forecast['yhat'].fillna(method='ffill').fillna(method='bfill')
        
        # Ensure reasonable bounds for predictions
        y_mean = forecast['yhat'].mean()
        y_std = forecast['yhat'].std()
        
        if y_std > 0:
            # Clip extreme outliers (beyond 5 standard deviations)
            lower_bound = y_mean - 5 * y_std
            upper_bound = y_mean + 5 * y_std
            
            outliers = ((forecast['yhat'] < lower_bound) | (forecast['yhat'] > upper_bound)).sum()
            if outliers > 0:
                print(f"[WARN] Clipping {outliers} extreme predictions")
                forecast['yhat'] = forecast['yhat'].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"[INFO] Prophet predictions completed: {len(forecast)} points")
        return forecast
        
    except Exception as e:
        print(f"[ERROR] Prophet prediction failed: {e}")
        return None

def validate_prophet_data(df: pd.DataFrame) -> bool:
    """
    Validate dataframe for Prophet compatibility.
    """
    try:
        if df is None or df.empty:
            return False
        
        if not all(col in df.columns for col in ['ds', 'y']):
            return False
        
        # Check data types
        try:
            pd.to_datetime(df['ds'])
            pd.to_numeric(df['y'])
        except (ValueError, TypeError):
            return False
        
        # Check for sufficient data
        if len(df) < 10:
            return False
        
        # Check for valid values
        if df['y'].isna().all():
            return False
        
        return True
        
    except Exception:
        return False
