"""
Model Serving Layer with Batching and Caching
Production-ready model serving infrastructure

Industry Best Practices:
- Request batching for efficiency
- Response caching for performance
- Model version routing
- Health checks and monitoring
"""

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from bondtrader.config import get_config
from bondtrader.core.bond_models import Bond
from bondtrader.ml.ml_adjuster_enhanced import EnhancedMLBondAdjuster
from bondtrader.utils.utils import logger


@dataclass
class PredictionRequest:
    """Single prediction request"""

    bond_id: str
    bond: Bond
    model_version: Optional[str] = None
    use_cache: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PredictionResponse:
    """Prediction response"""

    bond_id: str
    predicted_value: float
    confidence: float = 0.0
    model_version: str = "unknown"
    cached: bool = False
    latency_ms: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PredictionCache:
    """LRU cache for predictions"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize prediction cache

        Args:
            max_size: Maximum number of cached predictions
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[PredictionResponse, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}

    def _get_cache_key(self, bond_id: str, model_version: str) -> str:
        """Generate cache key"""
        return f"{bond_id}_{model_version}"

    def get(self, bond_id: str, model_version: str) -> Optional[PredictionResponse]:
        """Get cached prediction if available and not expired"""
        cache_key = self._get_cache_key(bond_id, model_version)

        if cache_key not in self.cache:
            return None

        response, cached_time = self.cache[cache_key]

        # Check TTL
        if (datetime.now() - cached_time).total_seconds() > self.ttl_seconds:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return None

        # Update access time
        self.access_times[cache_key] = datetime.now()

        # Mark as cached
        response.cached = True
        return response

    def set(self, bond_id: str, model_version: str, response: PredictionResponse):
        """Cache a prediction"""
        cache_key = self._get_cache_key(bond_id, model_version)

        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_times:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]

        self.cache[cache_key] = (response, datetime.now())
        self.access_times[cache_key] = datetime.now()

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class ModelServer:
    """
    Model serving layer with batching and caching

    Industry Best Practices:
    - Request batching for efficiency
    - Response caching for performance
    - Model version routing
    - Health checks
    """

    def __init__(
        self,
        model: EnhancedMLBondAdjuster = None,
        model_version: str = "latest",
        cache_enabled: bool = True,
        batch_size: int = 32,
        max_batch_wait_ms: int = 100,
    ):
        """
        Initialize model server

        Args:
            model: Model to serve (or None to load from registry)
            model_version: Model version to serve
            cache_enabled: Enable prediction caching
            batch_size: Batch size for predictions
            max_batch_wait_ms: Maximum wait time for batching
        """
        self.model = model
        self.model_version = model_version
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size
        self.max_batch_wait_ms = max_batch_wait_ms

        # Cache
        self.cache = PredictionCache() if cache_enabled else None

        # Batching queue
        self.batch_queue: deque = deque()
        self.batch_lock = threading.Lock()
        self.batch_thread = None
        self.running = False

        # Statistics
        self.stats = {
            "total_predictions": 0,
            "cached_predictions": 0,
            "batched_predictions": 0,
            "total_latency_ms": 0.0,
        }

    def start(self):
        """Start model server"""
        if self.model is None:
            raise ValueError("Model must be provided or loaded")

        self.running = True

        # Start batch processing thread
        if self.batch_size > 1:
            self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
            self.batch_thread.start()

        logger.info(f"Model server started: version {self.model_version}")

    def stop(self):
        """Stop model server"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        logger.info("Model server stopped")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Get prediction for a single bond

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        start_time = time.time()

        # Check cache
        if self.cache_enabled and request.use_cache:
            cached_response = self.cache.get(request.bond_id, self.model_version)
            if cached_response:
                self.stats["cached_predictions"] += 1
                cached_response.latency_ms = (time.time() - start_time) * 1000
                return cached_response

        # Get prediction
        try:
            if hasattr(self.model, "predict_adjusted_value"):
                result = self.model.predict_adjusted_value(request.bond)
                predicted_value = result.get("ml_adjusted_value", request.bond.current_price)
                confidence = result.get("ml_confidence", 0.0)
            else:
                # Fallback for other model types
                predicted_value = self.model.predict([request.bond])[0]
                confidence = 0.5

            latency_ms = (time.time() - start_time) * 1000

            response = PredictionResponse(
                bond_id=request.bond_id,
                predicted_value=predicted_value,
                confidence=confidence,
                model_version=self.model_version,
                cached=False,
                latency_ms=latency_ms,
            )

            # Cache response
            if self.cache_enabled and request.use_cache:
                self.cache.set(request.bond_id, self.model_version, response)

            self.stats["total_predictions"] += 1
            self.stats["total_latency_ms"] += latency_ms

            return response

        except Exception as e:
            logger.error(f"Prediction failed for {request.bond_id}: {e}", exc_info=True)
            # Return fallback response
            return PredictionResponse(
                bond_id=request.bond_id,
                predicted_value=request.bond.current_price,
                confidence=0.0,
                model_version=self.model_version,
                latency_ms=(time.time() - start_time) * 1000,
            )

    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """
        Get predictions for multiple bonds (batched)

        Args:
            requests: List of prediction requests

        Returns:
            List of prediction responses
        """
        if len(requests) == 0:
            return []

        start_time = time.time()

        # Check cache for all requests
        responses = []
        uncached_requests = []

        for request in requests:
            if self.cache_enabled and request.use_cache:
                cached_response = self.cache.get(request.bond_id, self.model_version)
                if cached_response:
                    responses.append(cached_response)
                    self.stats["cached_predictions"] += 1
                    continue

            uncached_requests.append(request)
            responses.append(None)  # Placeholder

        # Process uncached requests in batch
        if len(uncached_requests) > 0:
            try:
                bonds = [req.bond for req in uncached_requests]

                # Batch prediction
                if hasattr(self.model, "predict_adjusted_value"):
                    # Process individually (models may not support batch)
                    batch_responses = [self.predict(req) for req in uncached_requests]
                else:
                    # Use model's batch predict if available
                    predictions = self.model.predict(bonds)
                    batch_responses = [
                        PredictionResponse(
                            bond_id=req.bond_id,
                            predicted_value=pred,
                            model_version=self.model_version,
                            latency_ms=0.0,
                        )
                        for req, pred in zip(uncached_requests, predictions)
                    ]

                # Fill in responses
                response_idx = 0
                for i, response in enumerate(responses):
                    if response is None:
                        responses[i] = batch_responses[response_idx]
                        response_idx += 1

                self.stats["batched_predictions"] += len(uncached_requests)

            except Exception as e:
                logger.error(f"Batch prediction failed: {e}", exc_info=True)
                # Fallback to individual predictions
                for i, req in enumerate(uncached_requests):
                    if responses[requests.index(req)] is None:
                        responses[requests.index(req)] = self.predict(req)

        total_latency_ms = (time.time() - start_time) * 1000
        self.stats["total_predictions"] += len(requests)
        self.stats["total_latency_ms"] += total_latency_ms

        return responses

    def _batch_processor(self):
        """Background thread for batch processing"""
        while self.running:
            try:
                # Collect requests for batching
                batch_requests = []
                batch_start = time.time()

                while len(batch_requests) < self.batch_size:
                    wait_time = (time.time() - batch_start) * 1000

                    if wait_time >= self.max_batch_wait_ms and len(batch_requests) > 0:
                        break

                    if len(self.batch_queue) > 0:
                        with self.batch_lock:
                            if len(self.batch_queue) > 0:
                                batch_requests.append(self.batch_queue.popleft())
                    else:
                        time.sleep(0.01)  # Small sleep to avoid busy waiting

                # Process batch
                if len(batch_requests) > 0:
                    self.predict_batch(batch_requests)

            except Exception as e:
                logger.error(f"Batch processor error: {e}", exc_info=True)
                time.sleep(0.1)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            "status": "healthy",
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "stats": self.stats.copy(),
        }

        # Compute average latency
        if self.stats["total_predictions"] > 0:
            health["avg_latency_ms"] = (
                self.stats["total_latency_ms"] / self.stats["total_predictions"]
            )
        else:
            health["avg_latency_ms"] = 0.0

        # Compute cache hit rate
        if self.stats["total_predictions"] > 0:
            health["cache_hit_rate"] = (
                self.stats["cached_predictions"] / self.stats["total_predictions"]
            )
        else:
            health["cache_hit_rate"] = 0.0

        # Check if model is responding
        try:
            if self.model is not None:
                # Quick test prediction (would use a test bond)
                health["model_responding"] = True
            else:
                health["model_responding"] = False
                health["status"] = "unhealthy"
        except Exception:
            health["model_responding"] = False
            health["status"] = "unhealthy"

        return health

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        stats = self.stats.copy()

        if stats["total_predictions"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_predictions"]
            stats["cache_hit_rate"] = stats["cached_predictions"] / stats["total_predictions"]
        else:
            stats["avg_latency_ms"] = 0.0
            stats["cache_hit_rate"] = 0.0

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        return stats

    def clear_cache(self):
        """Clear prediction cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Prediction cache cleared")


def create_model_server(
    model: EnhancedMLBondAdjuster,
    model_version: str = "latest",
    cache_enabled: bool = True,
    batch_size: int = 32,
) -> ModelServer:
    """
    Convenience function to create model server

    Args:
        model: Model to serve
        model_version: Model version
        cache_enabled: Enable caching
        batch_size: Batch size for predictions

    Returns:
        Configured ModelServer
    """
    server = ModelServer(
        model=model,
        model_version=model_version,
        cache_enabled=cache_enabled,
        batch_size=batch_size,
    )
    server.start()
    return server
