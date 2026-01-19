"""
Feature Store Implementation
Centralized feature management for ML pipeline

Industry Best Practices:
- Feature versioning and lineage
- Online/offline feature serving
- Feature discovery and catalog
- Feature validation
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.utils.utils import logger


class FeatureStore:
    """
    Feature Store for centralized feature management

    Industry Best Practices:
    - Feature versioning
    - Feature lineage tracking
    - Online/offline serving
    - Feature catalog
    """

    def __init__(self, store_path: str = None):
        """
        Initialize feature store

        Args:
            store_path: Path to feature store directory
        """
        self.config = get_config()
        self.store_path = store_path or os.path.join(self.config.data_dir, "feature_store")
        os.makedirs(self.store_path, exist_ok=True)

        # Feature registry
        self.registry_path = os.path.join(self.store_path, "registry.json")
        self.registry = self._load_registry()

        # Feature cache for online serving
        self.cache = {}

    def _load_registry(self) -> Dict:
        """Load feature registry"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feature registry: {e}")
        return {}

    def _save_registry(self):
        """Save feature registry"""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature registry: {e}")

    def _compute_feature_hash(self, feature_data: np.ndarray) -> str:
        """Compute hash of feature data for versioning"""
        data_bytes = feature_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]

    def register_feature_set(
        self,
        feature_name: str,
        features: np.ndarray,
        feature_names: List[str],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Register a feature set in the store

        Args:
            feature_name: Name of the feature set
            features: Feature matrix
            feature_names: List of feature names
            metadata: Additional metadata

        Returns:
            Feature version ID
        """
        # Compute feature hash for versioning
        feature_hash = self._compute_feature_hash(features)
        version_id = f"{feature_name}_v{feature_hash}"

        # Create version directory
        version_path = os.path.join(self.store_path, version_id)
        os.makedirs(version_path, exist_ok=True)

        # Save features
        feature_file = os.path.join(version_path, "features.joblib")
        joblib.dump(features, feature_file)

        # Save metadata
        feature_metadata = {
            "feature_name": feature_name,
            "version_id": version_id,
            "feature_names": feature_names,
            "n_samples": len(features),
            "n_features": features.shape[1] if len(features.shape) > 1 else 1,
            "created_at": datetime.now().isoformat(),
            "feature_hash": feature_hash,
            **(metadata or {}),
        }

        metadata_file = os.path.join(version_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(feature_metadata, f, indent=2)

        # Update registry
        if feature_name not in self.registry:
            self.registry[feature_name] = {}

        self.registry[feature_name][version_id] = {
            "version_id": version_id,
            "created_at": feature_metadata["created_at"],
            "feature_hash": feature_hash,
            "metadata": feature_metadata,
        }

        # Set as latest if first version
        if "latest" not in self.registry[feature_name]:
            self.registry[feature_name]["latest"] = version_id

        self._save_registry()

        logger.info(f"Registered feature set: {feature_name} version {version_id}")
        return version_id

    def get_feature_set(self, feature_name: str, version: str = "latest") -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Get feature set from store

        Args:
            feature_name: Name of feature set
            version: Version to retrieve (default: "latest")

        Returns:
            Tuple of (features, feature_names, metadata)
        """
        if feature_name not in self.registry:
            raise ValueError(f"Feature set {feature_name} not found in registry")

        # Get version ID
        if version == "latest":
            version_id = self.registry[feature_name].get("latest")
            if version_id is None:
                # Use most recent version
                versions = [v for k, v in self.registry[feature_name].items() if k != "latest"]
                if not versions:
                    raise ValueError(f"No versions found for {feature_name}")
                version_id = max(versions, key=lambda x: x["created_at"])["version_id"]
        else:
            version_id = version

        if version_id not in self.registry[feature_name]:
            raise ValueError(f"Version {version_id} not found for {feature_name}")

        # Load features
        version_path = os.path.join(self.store_path, version_id)
        feature_file = os.path.join(version_path, "features.joblib")

        if not os.path.exists(feature_file):
            raise ValueError(f"Feature file not found: {feature_file}")

        features = joblib.load(feature_file)

        # Load metadata
        metadata_file = os.path.join(version_path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        feature_names = metadata["feature_names"]

        return features, feature_names, metadata

    def list_feature_sets(self) -> List[str]:
        """List all registered feature sets"""
        return list(self.registry.keys())

    def list_versions(self, feature_name: str) -> List[str]:
        """List all versions of a feature set"""
        if feature_name not in self.registry:
            return []

        return [v["version_id"] for k, v in self.registry[feature_name].items() if k != "latest"]

    def get_feature_lineage(self, feature_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Get feature lineage (dependencies, transformations)

        Args:
            feature_name: Name of feature set
            version: Version to get lineage for

        Returns:
            Lineage information
        """
        _, _, metadata = self.get_feature_set(feature_name, version)

        lineage = {
            "feature_name": feature_name,
            "version": version,
            "created_at": metadata.get("created_at"),
            "dependencies": metadata.get("dependencies", []),
            "transformations": metadata.get("transformations", []),
            "data_source": metadata.get("data_source"),
        }

        return lineage

    def serve_features_online(self, bond_ids: List[str], feature_name: str, version: str = "latest") -> pd.DataFrame:
        """
        Serve features for online prediction (cached)

        Args:
            bond_ids: List of bond IDs to get features for
            feature_name: Name of feature set
            version: Version to use

        Returns:
            DataFrame with features indexed by bond_id
        """
        # Check cache first
        cache_key = f"{feature_name}_{version}"
        if cache_key in self.cache:
            features_df = self.cache[cache_key]
            # Return subset for requested bond_ids
            if isinstance(features_df, pd.DataFrame) and "bond_id" in features_df.columns:
                return features_df[features_df["bond_id"].isin(bond_ids)]

        # Load features
        features, feature_names, metadata = self.get_feature_set(feature_name, version)

        # Create DataFrame (assuming bond_ids match feature order)
        # In production, this would be indexed by bond_id
        features_df = pd.DataFrame(features, columns=feature_names)
        if len(bond_ids) == len(features):
            features_df["bond_id"] = bond_ids

        # Cache for future use
        self.cache[cache_key] = features_df

        return features_df

    def compute_features(
        self,
        bonds: List[Bond],
        feature_compute_fn,
        feature_name: str,
        metadata: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute and register features

        Args:
            bonds: List of bonds
            feature_compute_fn: Function to compute features (bonds -> features, feature_names)
            feature_name: Name for feature set
            metadata: Additional metadata

        Returns:
            Tuple of (features, feature_names)
        """
        # Compute features
        features, feature_names = feature_compute_fn(bonds)

        # Add lineage metadata
        if metadata is None:
            metadata = {}

        metadata["dependencies"] = metadata.get("dependencies", [])
        metadata["transformations"] = metadata.get("transformations", [])
        metadata["data_source"] = metadata.get("data_source", "bonds")
        metadata["n_bonds"] = len(bonds)

        # Register in store
        version_id = self.register_feature_set(feature_name, features, feature_names, metadata)

        logger.info(f"Computed and registered features: {feature_name} version {version_id}")

        return features, feature_names


# Global feature store instance
_feature_store_instance: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get global feature store instance (singleton)"""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore()
    return _feature_store_instance
