"""
Data Lineage Tracking for Reproducibility
Full data lineage tracking for ML pipeline

Industry Best Practices:
- Dataset versioning
- Feature lineage tracking
- Model-to-data lineage
- Full reproducibility metadata
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from bondtrader.config import get_config
from bondtrader.utils.utils import logger


class LineageType(Enum):
    """Types of lineage relationships"""

    DATASET_TO_FEATURE = "dataset_to_feature"
    FEATURE_TO_MODEL = "feature_to_model"
    MODEL_TO_PREDICTION = "model_to_prediction"
    DATA_TRANSFORMATION = "data_transformation"


@dataclass
class DatasetVersion:
    """Versioned dataset information"""

    dataset_id: str
    version: str
    data_hash: str
    file_path: str
    created_at: datetime
    n_samples: int
    n_features: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureLineage:
    """Feature lineage information"""

    feature_name: str
    feature_version: str
    source_datasets: List[str]  # Dataset IDs
    transformations: List[str]  # Transformation steps
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelLineage:
    """Model lineage information"""

    model_name: str
    model_version: str
    feature_versions: List[str]  # Feature version IDs
    dataset_versions: List[str]  # Dataset version IDs
    training_config: Dict[str, Any]
    created_at: datetime
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageNode:
    """Node in lineage graph"""

    node_id: str
    node_type: str  # dataset, feature, model, prediction
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """Edge in lineage graph"""

    source_id: str
    target_id: str
    edge_type: LineageType
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLineageTracker:
    """
    Data lineage tracking for full reproducibility

    Industry Best Practices:
    - Dataset versioning
    - Feature lineage
    - Model-to-data lineage
    - Full audit trail
    """

    def __init__(self, lineage_dir: str = None):
        """
        Initialize data lineage tracker

        Args:
            lineage_dir: Directory to store lineage data
        """
        self.config = get_config()
        self.lineage_dir = lineage_dir or os.path.join(self.config.data_dir, "lineage")
        os.makedirs(self.lineage_dir, exist_ok=True)

        # Lineage registry
        self.registry_file = os.path.join(self.lineage_dir, "lineage_registry.json")
        self.registry = self._load_registry()

        # Lineage graph
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []

    def _load_registry(self) -> Dict:
        """Load lineage registry"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load lineage registry: {e}")
        return {
            "datasets": {},
            "features": {},
            "models": {},
        }

    def _save_registry(self):
        """Save lineage registry"""
        try:
            # Convert datetime objects to strings
            registry_copy = json.loads(json.dumps(self.registry, default=str))
            with open(self.registry_file, "w") as f:
                json.dump(registry_copy, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save lineage registry: {e}")

    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for versioning"""
        if isinstance(data, (list, tuple)):
            data_str = str(sorted(data))
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def register_dataset(
        self,
        dataset_id: str,
        file_path: str,
        n_samples: int,
        n_features: int,
        metadata: Dict[str, Any] = None,
    ) -> DatasetVersion:
        """
        Register a dataset version

        Args:
            dataset_id: Unique dataset identifier
            file_path: Path to dataset file
            n_samples: Number of samples
            n_features: Number of features
            metadata: Additional metadata

        Returns:
            DatasetVersion object
        """
        # Compute data hash
        try:
            import os

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            else:
                file_hash = self._compute_data_hash(f"{dataset_id}_{n_samples}_{n_features}")
        except Exception:
            file_hash = self._compute_data_hash(f"{dataset_id}_{n_samples}_{n_features}")

        version = f"v{file_hash}"

        dataset_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            data_hash=file_hash,
            file_path=file_path,
            created_at=datetime.now(),
            n_samples=n_samples,
            n_features=n_features,
            metadata=metadata or {},
        )

        # Register in registry
        if dataset_id not in self.registry["datasets"]:
            self.registry["datasets"][dataset_id] = {}

        self.registry["datasets"][dataset_id][version] = asdict(dataset_version)
        self.registry["datasets"][dataset_id][version][
            "created_at"
        ] = dataset_version.created_at.isoformat()

        # Add to lineage graph
        node_id = f"dataset_{dataset_id}_{version}"
        self.nodes[node_id] = LineageNode(
            node_id=node_id,
            node_type="dataset",
            version=version,
            metadata=metadata or {},
        )

        self._save_registry()

        logger.info(f"Registered dataset: {dataset_id} version {version}")

        return dataset_version

    def register_feature(
        self,
        feature_name: str,
        feature_version: str,
        source_datasets: List[str],
        transformations: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> FeatureLineage:
        """
        Register feature lineage

        Args:
            feature_name: Feature name
            feature_version: Feature version
            source_datasets: List of source dataset IDs
            transformations: List of transformation steps
            metadata: Additional metadata

        Returns:
            FeatureLineage object
        """
        feature_lineage = FeatureLineage(
            feature_name=feature_name,
            feature_version=feature_version,
            source_datasets=source_datasets,
            transformations=transformations or [],
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        # Register in registry
        feature_key = f"{feature_name}_{feature_version}"
        if feature_key not in self.registry["features"]:
            self.registry["features"][feature_key] = asdict(feature_lineage)
            self.registry["features"][feature_key][
                "created_at"
            ] = feature_lineage.created_at.isoformat()

        # Add to lineage graph
        node_id = f"feature_{feature_name}_{feature_version}"
        self.nodes[node_id] = LineageNode(
            node_id=node_id,
            node_type="feature",
            version=feature_version,
            metadata=metadata or {},
        )

        # Add edges from datasets to feature
        for dataset_id in source_datasets:
            # Find latest dataset version
            if dataset_id in self.registry["datasets"]:
                dataset_versions = self.registry["datasets"][dataset_id]
                if dataset_versions:
                    latest_version = max(dataset_versions.keys())
                    source_node_id = f"dataset_{dataset_id}_{latest_version}"

                    if source_node_id in self.nodes:
                        self.edges.append(
                            LineageEdge(
                                source_id=source_node_id,
                                target_id=node_id,
                                edge_type=LineageType.DATASET_TO_FEATURE,
                                metadata={"transformation": "feature_extraction"},
                            )
                        )

        self._save_registry()

        logger.info(f"Registered feature lineage: {feature_name} version {feature_version}")

        return feature_lineage

    def register_model(
        self,
        model_name: str,
        model_version: str,
        feature_versions: List[str],
        dataset_versions: List[str] = None,
        training_config: Dict[str, Any] = None,
        git_commit: str = None,
        metadata: Dict[str, Any] = None,
    ) -> ModelLineage:
        """
        Register model lineage

        Args:
            model_name: Model name
            model_version: Model version
            feature_versions: List of feature versions used
            dataset_versions: List of dataset versions used
            training_config: Training configuration
            git_commit: Git commit hash
            metadata: Additional metadata

        Returns:
            ModelLineage object
        """
        model_lineage = ModelLineage(
            model_name=model_name,
            model_version=model_version,
            feature_versions=feature_versions,
            dataset_versions=dataset_versions or [],
            training_config=training_config or {},
            created_at=datetime.now(),
            git_commit=git_commit,
            metadata=metadata or {},
        )

        # Register in registry
        model_key = f"{model_name}_{model_version}"
        if model_key not in self.registry["models"]:
            self.registry["models"][model_key] = asdict(model_lineage)
            self.registry["models"][model_key]["created_at"] = model_lineage.created_at.isoformat()

        # Add to lineage graph
        node_id = f"model_{model_name}_{model_version}"
        self.nodes[node_id] = LineageNode(
            node_id=node_id,
            node_type="model",
            version=model_version,
            metadata=metadata or {},
        )

        # Add edges from features to model
        for feature_version in feature_versions:
            # Find feature node
            feature_node_id = None
            for node_id_key, node in self.nodes.items():
                if node.node_type == "feature" and feature_version in node_id_key:
                    feature_node_id = node_id_key
                    break

            if feature_node_id:
                self.edges.append(
                    LineageEdge(
                        source_id=feature_node_id,
                        target_id=node_id,
                        edge_type=LineageType.FEATURE_TO_MODEL,
                        metadata={"training": "model_training"},
                    )
                )

        self._save_registry()

        logger.info(f"Registered model lineage: {model_name} version {model_version}")

        return model_lineage

    def get_lineage(
        self,
        node_id: str,
        direction: str = "both",  # upstream, downstream, both
    ) -> Dict[str, Any]:
        """
        Get lineage for a node

        Args:
            node_id: Node identifier
            direction: Direction to traverse (upstream, downstream, both)

        Returns:
            Lineage graph dictionary
        """
        if node_id not in self.nodes:
            return {"error": f"Node {node_id} not found"}

        upstream_nodes = []
        downstream_nodes = []

        # Find upstream nodes (sources)
        if direction in ["upstream", "both"]:
            for edge in self.edges:
                if edge.target_id == node_id:
                    upstream_nodes.append(
                        {
                            "node_id": edge.source_id,
                            "edge_type": edge.edge_type.value,
                            "metadata": edge.metadata,
                        }
                    )

        # Find downstream nodes (targets)
        if direction in ["downstream", "both"]:
            for edge in self.edges:
                if edge.source_id == node_id:
                    downstream_nodes.append(
                        {
                            "node_id": edge.target_id,
                            "edge_type": edge.edge_type.value,
                            "metadata": edge.metadata,
                        }
                    )

        return {
            "node": asdict(self.nodes[node_id]),
            "upstream": upstream_nodes,
            "downstream": downstream_nodes,
        }

    def get_full_lineage(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        Get full lineage from datasets to model

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            Complete lineage graph
        """
        model_key = f"{model_name}_{model_version}"
        if model_key not in self.registry["models"]:
            return {"error": f"Model {model_key} not found"}

        model_info = self.registry["models"][model_key]

        # Get all upstream nodes
        model_node_id = f"model_{model_name}_{model_version}"
        lineage = self.get_lineage(model_node_id, direction="upstream")

        # Add model information
        lineage["model"] = model_info

        return lineage

    def export_lineage(self, output_path: str):
        """Export lineage to file"""
        export_data = {
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
            "registry": self.registry,
        }

        # Convert datetime objects
        export_str = json.dumps(export_data, default=str, indent=2)

        with open(output_path, "w") as f:
            f.write(export_str)

        logger.info(f"Exported lineage to {output_path}")


# Global lineage tracker instance
_lineage_tracker_instance: Optional[DataLineageTracker] = None


def get_lineage_tracker() -> DataLineageTracker:
    """Get global lineage tracker instance (singleton)"""
    global _lineage_tracker_instance
    if _lineage_tracker_instance is None:
        _lineage_tracker_instance = DataLineageTracker()
    return _lineage_tracker_instance
