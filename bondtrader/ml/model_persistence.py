"""
Unified Model Persistence Utility
Centralizes save/load logic to eliminate duplication across ML modules
"""

import os
from typing import Any, Dict

import joblib

from bondtrader.utils.utils import logger
from bondtrader.utils.validation import validate_file_path


class ModelPersistence:
    """Unified model save/load with atomic writes"""

    @staticmethod
    def save_model(model_data: Dict[str, Any], filepath: str) -> None:
        """
        Save model data with atomic writes to prevent corruption

        Args:
            model_data: Dictionary containing model, scaler, and metadata
            filepath: Path to save model (relative path recommended)

        Raises:
            ValueError: If filepath is invalid
            OSError: If file I/O fails
        """
        # Validate and sanitize file path
        validate_file_path(filepath, allow_absolute=False, allowed_extensions=[".joblib", ".pkl", ".model"], name="filepath")

        # Create directory if needed
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(dir_path, exist_ok=True)

        # Save to temporary file first (atomic on Unix, best-effort on Windows)
        temp_filepath = filepath + ".tmp"
        try:
            joblib.dump(model_data, temp_filepath)

            # Atomic rename (Unix) or copy+remove (Windows fallback)
            if os.name == "nt":  # Windows
                # On Windows, rename might fail if file exists, so remove first
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
            else:  # Unix/Linux/Mac
                os.rename(temp_filepath, filepath)

        except (OSError, IOError, PermissionError) as e:
            # Clean up temp file on file I/O errors
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Ignore cleanup errors
            raise
        except Exception as e:
            # Clean up temp file on unexpected errors
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Ignore cleanup errors
            logger.error(f"Unexpected error saving model to {filepath}: {e}", exc_info=True)
            raise

    @staticmethod
    def load_model(filepath: str, required_keys: list = None) -> Dict[str, Any]:
        """
        Load model data with validation

        Args:
            filepath: Path to saved model file
            required_keys: List of required keys in model data (default: ["model", "scaler"])

        Returns:
            Dictionary containing model data

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or missing required keys
            TypeError: If model file contains invalid data types
        """
        if required_keys is None:
            required_keys = ["model", "scaler"]

        # Validate and sanitize file path
        validate_file_path(
            filepath, must_exist=True, allow_absolute=False, allowed_extensions=[".joblib", ".pkl", ".model"], name="filepath"
        )

        try:
            data = joblib.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {e}") from e

        # Validate required keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Model file missing required keys: {missing_keys}")

        return data
