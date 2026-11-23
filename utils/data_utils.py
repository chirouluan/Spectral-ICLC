"""
Spectral data utilities.
Handles 3D (n_samples, n_pixels, n_bands) <-> 2D (n_samples, n_bands) adaptation,
data validation, missing value handling, and outlier detection.

Ported from TimeSeriesScientist spectral_data_utils.py, adapted for 3D data.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 3D <-> 2D Adapters
# ============================================================

def squeeze_pixels(data: np.ndarray) -> np.ndarray:
    """
    Convert (n_samples, n_pixels, n_bands) -> (n_samples, n_bands).
    Only works when n_pixels == 1. Raises if n_pixels > 1.
    """
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    n_samples, n_pixels, n_bands = data.shape
    if n_pixels == 1:
        return data.reshape(n_samples, n_bands)
    raise ValueError(
        f"n_pixels={n_pixels} > 1. Use aggregate_pixels() first to reduce to single pixel."
    )


def unsqueeze_pixels(X_2d: np.ndarray) -> np.ndarray:
    """Convert (n_samples, n_bands) -> (n_samples, 1, n_bands)."""
    if X_2d.ndim == 3:
        return X_2d
    return X_2d.reshape(X_2d.shape[0], 1, X_2d.shape[1])


def aggregate_pixels(data: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Reduce (n_samples, n_pixels, n_bands) -> (n_samples, n_bands) by aggregating pixels.

    Args:
        data: (n_samples, n_pixels, n_bands)
        method: "mean", "median", or "center"

    Returns:
        (n_samples, n_bands)
    """
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    if method == "mean":
        return np.mean(data, axis=1)
    elif method == "median":
        return np.median(data, axis=1)
    elif method == "center":
        mid = data.shape[1] // 2
        return data[:, mid, :]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ============================================================
# Missing Value Handling (from TSS SpectralPreprocessor)
# ============================================================

def handle_missing_values(X: np.ndarray, strategy: str = "interpolate") -> np.ndarray:
    """
    Handle missing values in spectral data.

    Args:
        X: spectral matrix, 2D (n_samples, n_wavelengths) or 3D (n_samples, n_pixels, n_bands)
        strategy: "interpolate", "mean", "median", "drop", "none"

    Returns:
        Processed array. For "drop", rows with NaN are removed (may change first dimension).
    """
    original_shape = X.shape
    # Flatten 3D -> 2D: (n_samples, n_pixels, n_bands) -> (n_samples*n_pixels, n_bands)
    if X.ndim == 3:
        n_samples, n_pixels, n_bands = X.shape
        X_2d = X.reshape(n_samples * n_pixels, n_bands)
    else:
        X_2d = X

    if strategy == "none" or not np.any(np.isnan(X_2d)):
        return X

    X_work = X_2d.copy()

    if strategy == "mean":
        col_mean = np.nanmean(X_work, axis=0)
        indices = np.where(np.isnan(X_work))
        X_work[indices] = np.take(col_mean, indices[1])

    elif strategy == "median":
        col_median = np.nanmedian(X_work, axis=0)
        indices = np.where(np.isnan(X_work))
        X_work[indices] = np.take(col_median, indices[1])

    elif strategy == "interpolate":
        for i in range(X_work.shape[0]):
            mask = np.isnan(X_work[i, :])
            if mask.any():
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 0:
                    X_work[i, mask] = np.interp(
                        np.where(mask)[0], valid_indices, X_work[i, valid_indices]
                    )

    elif strategy == "drop":
        mask = ~np.any(np.isnan(X_work), axis=1)
        X_work = X_work[mask]
        logger.info(f"Dropped {np.sum(~mask)} samples with NaN values")
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    if X.ndim == 3:
        return X_work.reshape(original_shape)
    return X_work


# ============================================================
# Outlier Detection (from TSS SpectralPreprocessor)
# ============================================================

def detect_outliers(
    X: np.ndarray, method: str = "zscore", threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outlier samples.

    Args:
        X: 2D (n_samples, n_wavelengths) or 3D
        method: "zscore", "iqr", or "isolation_forest"
        threshold: threshold for zscore/iqr

    Returns:
        Boolean array, True = normal sample, False = outlier.
    """
    X_2d = squeeze_pixels(X) if X.ndim == 3 else X

    if method == "zscore":
        from scipy.stats import zscore as sp_zscore
        z_scores = np.abs(sp_zscore(X_2d, axis=1))
        mean_z = np.mean(z_scores, axis=1)
        return mean_z < threshold

    elif method == "iqr":
        Q1 = np.percentile(X_2d, 25, axis=0)
        Q3 = np.percentile(X_2d, 75, axis=0)
        IQR = Q3 - Q1
        IQR[IQR == 0] = 1e-10
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outlier_mask = np.any((X_2d < lower) | (X_2d > upper), axis=1)
        return ~outlier_mask

    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(X_2d)
        return predictions == 1

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# ============================================================
# Extreme Value Clipping (from TSS SpectralPreprocessor)
# ============================================================

def clip_extreme_values(X: np.ndarray, n_mads: float = 5.0) -> np.ndarray:
    """
    Clip extreme values based on Median Absolute Deviation.

    Args:
        X: 2D (n_samples, n_wavelengths) or 3D
        n_mads: MAD multiplier threshold

    Returns:
        Clipped array (same shape).
    """
    original_shape = X.shape
    X_2d = squeeze_pixels(X) if X.ndim == 3 else X

    median_vals = np.median(X_2d, axis=0)
    mad_vals = np.median(np.abs(X_2d - median_vals), axis=0)
    mad_vals[mad_vals == 0] = 1e-10

    lower = median_vals - n_mads * 1.4826 * mad_vals
    upper = median_vals + n_mads * 1.4826 * mad_vals

    X_clipped = np.clip(X_2d, lower, upper)

    n_clipped = np.sum(X_2d != X_clipped)
    if n_clipped > 0:
        logger.info(f"Clipped {n_clipped} extreme values ({n_clipped / X_2d.size * 100:.2f}%)")

    if X.ndim == 3:
        return X_clipped.reshape(original_shape)
    return X_clipped


# ============================================================
# Data Validation
# ============================================================

def validate_spectral_data(
    data: np.ndarray,
    y: Optional[np.ndarray] = None,
    task_type: str = "quantitative",
) -> Dict[str, Any]:
    """
    Validate spectral data quality.

    Args:
        data: raw spectral data, 2D or 3D numpy array
        y: target variable (optional)
        task_type: "quantitative", "classification", or "anomaly detection"

    Returns:
        Dict with keys: is_valid (bool), errors (list), warnings (list), info (dict)
    """
    errors = []
    warnings = []

    # --- Shape checks ---
    if data.ndim == 3:
        n_samples, n_pixels, n_bands = data.shape
        if n_pixels > 1:
            warnings.append(
                f"n_pixels={n_pixels} > 1. Consider using aggregate_pixels()."
            )
    elif data.ndim == 2:
        n_samples, n_bands = data.shape
    else:
        errors.append(f"Expected 2D or 3D array, got {data.ndim}D")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "info": {}}

    # --- NaN check ---
    nan_count = np.sum(np.isnan(data))
    nan_ratio = nan_count / data.size
    if nan_ratio > 0.1:
        errors.append(f"High NaN ratio: {nan_ratio:.2%} ({nan_count} values)")
    elif nan_ratio > 0:
        warnings.append(f"Contains {nan_count} NaN values ({nan_ratio:.2%})")

    # --- Inf check ---
    inf_count = np.sum(np.isinf(data))
    if inf_count > 0:
        errors.append(f"Contains {inf_count} infinite values")

    # --- Target variable checks ---
    if y is not None:
        y_arr = np.asarray(y)
        y_len = y_arr.size if y_arr.ndim >= 2 else len(y_arr)
        total_samples = n_samples * (data.shape[1] if data.ndim == 3 else 1)
        if y_len != n_samples and y_len != total_samples:
            errors.append(f"y length ({y_len}) != n_samples ({n_samples}) and != total samples ({total_samples})")

        if task_type == "quantitative":
            if not np.issubdtype(np.asarray(y).dtype, np.number):
                errors.append("Target must be numeric for quantitative tasks")
            elif np.any(np.isnan(np.asarray(y, dtype=float))):
                warnings.append("Target variable contains NaN")
        elif task_type == "classification":
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                errors.append("Classification target must have >= 2 classes")
            if len(unique_classes) > 50:
                warnings.append(f"Large number of classes: {len(unique_classes)}")

    # --- Sample count ---
    if n_samples < 30:
        warnings.append(f"Small sample count: {n_samples}")

    # --- Outlier estimate (only if no NaN) ---
    outlier_ratio = 0.0
    if nan_count == 0 and n_samples > 5:
        try:
            from scipy.stats import zscore as sp_zscore
            X_2d = squeeze_pixels(data) if data.ndim == 3 else data
            z_scores = np.abs(sp_zscore(X_2d, axis=1))
            mean_z = np.mean(z_scores, axis=1)
            outlier_ratio = float(np.mean(mean_z >= 3.0))
            if outlier_ratio > 0.05:
                warnings.append(f"Potential outlier ratio: {outlier_ratio:.2%}")
        except Exception:
            pass

    info = {
        "n_samples": n_samples,
        "n_bands": n_bands,
        "nan_ratio": float(nan_ratio),
        "inf_count": int(inf_count),
        "outlier_ratio": float(outlier_ratio),
    }
    if data.ndim == 3:
        info["n_pixels"] = data.shape[1]

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }
