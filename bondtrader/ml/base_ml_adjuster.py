"""
Base ML Bond Adjuster Class
Provides common functionality for all ML adjuster implementations
Reduces code duplication and ensures consistent behavior
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger
from bondtrader.utils.validation import validate_file_path


class BaseMLBondAdjuster(ABC):
    """
    Base class for ML bond adjusters
    Provides common functionality: save_model, load_model, validation
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize base ML adjuster

        Args:
            model_type: ML model type (random_forest, gradient_boosting, etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.valuator = BondValuator()

    def save_model(self, filepath: str) -> None:
        """
        Save trained model and scaler with atomic writes and path validation

        Args:
            filepath: Path to save model (relative path recommended)

        Raises:
            ValueError: If model not trained or filepath invalid
            OSError: If file I/O fails
            PermissionError: If insufficient permissions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Validate and sanitize file path
        validate_file_path(
            filepath,
            allow_absolute=False,
            allowed_extensions=['.joblib', '.pkl', '.model'],
            name="filepath"
        )

        # Create directory if needed
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(dir_path, exist_ok=True)

        # Get model data to save (implemented by subclasses)
        model_data = self._get_model_data()

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

    def load_model(self, filepath: str) -> None:
        """
        Load trained model and scaler

        Args:
            filepath: Path to saved model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or missing required keys
            TypeError: If model file contains invalid data types
        """
        # Validate and sanitize file path
        validate_file_path(
            filepath,
            must_exist=True,
            allow_absolute=False,
            allowed_extensions=['.joblib', '.pkl', '.model'],
            name="filepath"
        )

        try:
            data = joblib.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {e}") from e

        # Validate required keys (minimum required by base class)
        required_keys = ["model", "scaler"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Model file missing required keys: {missing_keys}")

        try:
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.model_type = data.get("model_type", self.model_type)
            self.is_trained = True

            # Load additional data (implemented by subclasses)
            self._load_additional_data(data)

        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"Invalid model data structure: {e}") from e

    @abstractmethod
    def _get_model_data(self) -> Dict:
        """
        Get model data to save (implemented by subclasses)

        Returns:
            Dictionary with model data to save
        """
        pass

    def _load_additional_data(self, data: Dict) -> None:
        """
        Load additional model-specific data (can be overridden by subclasses)

        Args:
            data: Dictionary with loaded model data
        """
        # Default: no additional data to load
        pass

    @abstractmethod
    def train(self, bonds: List[Bond], **kwargs) -> Dict:
        """
        Train the ML model (must be implemented by subclasses)

        Args:
            bonds: List of bonds to train on
            **kwargs: Additional training parameters

        Returns:
            Training metrics dictionary
        """
        pass

    @abstractmethod
    def predict_adjusted_value(self, bond: Bond) -> Dict:
        """
        Predict adjusted value for a bond (must be implemented by subclasses)

        Args:
            bond: Bond to predict for

        Returns:
            Dictionary with prediction results
        """
        pass

    def _create_base_features(self, bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """
        Create base feature matrix (common across ML adjusters)

        Args:
            bonds: List of bonds
            fair_values: List of fair values for bonds

        Returns:
            Feature matrix as numpy array
        """
        features = []
        for bond, fv in zip(bonds, fair_values):
            char = bond.get_bond_characteristics()

            # Base characteristics
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
            ]

            # Additional derived features
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            # Price relative to fair value
            price_to_fair_ratio = bond.current_price / fv if fv > 0 else 1.0

            feature_vector.extend(
                [
                    ytm * 100,  # YTM as percentage
                    duration,
                    convexity,
                    price_to_fair_ratio,
                    bond.face_value,  # Size factor
                ]
            )

            features.append(feature_vector)

        return np.array(features)
