#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector database for LLaDA Memory System.

This module provides a persistent vector database for memory storage.
"""

import json
import logging
import os
import time
from typing import List, Dict, Tuple, Any

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class MemoryVectorDB:
    """Simple vector database for memory storage."""

    def __init__(self, config_path=None):
        """Initialize the vector database.
        
        Args:
            config_path: Path to configuration file
        """
        # Default configuration
        self.config = {
            "vector_db_path": os.path.join(os.getcwd(), "data", "memory", "vector_db"),
            "dimension": 64,
            "use_vector_db": True,
            "similarity_threshold": 0.7,
            "max_vectors": 1000,
            "pruning_strategy": "lru",  # Least recently used
        }

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                logger.error(f"Error loading vector DB config: {e}")

        # Ensure vector DB path exists
        os.makedirs(self.config["vector_db_path"], exist_ok=True)

        # Initialize vectors storage
        self.vectors = []
        self.metadata = []
        self.usage_info = []

        # Load existing vectors if any
        self._load_vectors()

    def _load_vectors(self):
        """Load existing vectors from disk."""
        try:
            vectors_file = os.path.join(self.config["vector_db_path"], "vectors.npy")
            metadata_file = os.path.join(self.config["vector_db_path"], "metadata.json")

            if os.path.exists(vectors_file) and os.path.exists(metadata_file):
                # Load vectors
                self.vectors = np.load(vectors_file)

                # Load metadata
                with open(metadata_file, 'r') as f:
                    loaded_data = json.load(f)
                    self.metadata = loaded_data.get("metadata", [])
                    self.usage_info = loaded_data.get("usage_info", [])

                logger.info(f"Loaded {len(self.vectors)} vectors from disk")

                # Initialize usage info if not present
                if not self.usage_info or len(self.usage_info) != len(self.vectors):
                    self.usage_info = [{"last_accessed": time.time(), "access_count": 0} for _ in
                                       range(len(self.vectors))]
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
            # Initialize empty vectors
            self.vectors = np.zeros((0, self.config["dimension"]))
            self.metadata = []
            self.usage_info = []

    def _save_vectors(self):
        """Save vectors to disk."""
        try:
            vectors_file = os.path.join(self.config["vector_db_path"], "vectors.npy")
            metadata_file = os.path.join(self.config["vector_db_path"], "metadata.json")

            # Save vectors
            np.save(vectors_file, self.vectors)

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump({
                    "metadata": self.metadata,
                    "usage_info": self.usage_info
                }, f)

            logger.info(f"Saved {len(self.vectors)} vectors to disk")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")

    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any] = None) -> int:
        """Add a vector to the database.
        
        Args:
            vector: Vector to add
            metadata: Optional metadata
            
        Returns:
            Index of the added vector
        """
        # Check vector dimension
        if vector.shape[0] != self.config["dimension"]:
            raise ValueError(f"Vector dimension mismatch: {vector.shape[0]} != {self.config['dimension']}")

        # Check if similar vector already exists
        similar_idx = self.find_similar(vector, top_k=1, threshold=self.config["similarity_threshold"])
        if similar_idx:
            # Update existing vector
            idx = similar_idx[0]
            self.vectors[idx] = vector
            if metadata:
                self.metadata[idx].update(metadata)

            # Update usage info
            self.usage_info[idx]["last_accessed"] = time.time()
            self.usage_info[idx]["access_count"] += 1

            logger.info(f"Updated similar vector at index {idx}")

            # Save changes
            self._save_vectors()

            return idx

        # Check if we need to prune vectors
        if len(self.vectors) >= self.config["max_vectors"]:
            self._prune_vectors()

        # Convert vectors to list if empty
        if len(self.vectors) == 0:
            self.vectors = np.zeros((0, self.config["dimension"]))

        # Add vector
        self.vectors = np.vstack([self.vectors, vector])
        self.metadata.append(metadata or {})
        self.usage_info.append({
            "last_accessed": time.time(),
            "access_count": 1
        })

        idx = len(self.vectors) - 1
        logger.info(f"Added new vector at index {idx}")

        # Save changes
        self._save_vectors()

        return idx

    def _prune_vectors(self):
        """Prune vectors based on pruning strategy."""
        if not self.vectors.any():
            return

        strategy = self.config["pruning_strategy"]

        if strategy == "lru":
            # Remove least recently used vector
            last_accessed = [info["last_accessed"] for info in self.usage_info]
            idx_to_remove = np.argmin(last_accessed)
        elif strategy == "lfu":
            # Remove least frequently used vector
            access_counts = [info["access_count"] for info in self.usage_info]
            idx_to_remove = np.argmin(access_counts)
        else:
            # Default: remove oldest vector
            idx_to_remove = 0

        # Remove vector
        self.vectors = np.delete(self.vectors, idx_to_remove, axis=0)
        self.metadata.pop(idx_to_remove)
        self.usage_info.pop(idx_to_remove)

        logger.info(f"Pruned vector at index {idx_to_remove}")

    def find_similar(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = None) -> List[int]:
        """Find similar vectors in the database.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of indices of similar vectors
        """
        if not self.vectors.any():
            return []

        # Check vector dimension
        if query_vector.shape[0] != self.config["dimension"]:
            raise ValueError(f"Query vector dimension mismatch: {query_vector.shape[0]} != {self.config['dimension']}")

        # Calculate cosine similarity
        norm_query = np.linalg.norm(query_vector)
        norm_vectors = np.linalg.norm(self.vectors, axis=1)

        # Avoid division by zero
        if norm_query == 0 or np.any(norm_vectors == 0):
            return []

        # Calculate similarities
        similarities = np.dot(self.vectors, query_vector) / (norm_vectors * norm_query)

        # Apply threshold if provided
        if threshold is not None:
            mask = similarities >= threshold
            if not np.any(mask):
                return []

            # Get indices of vectors above threshold
            indices = np.where(mask)[0]

            # Sort by similarity
            sorted_indices = indices[np.argsort(-similarities[indices])]

            # Return top_k
            result = sorted_indices[:top_k].tolist()
        else:
            # Sort by similarity
            sorted_indices = np.argsort(-similarities)

            # Return top_k
            result = sorted_indices[:top_k].tolist()

        # Update usage info
        for idx in result:
            self.usage_info[idx]["last_accessed"] = time.time()
            self.usage_info[idx]["access_count"] += 1

        return result

    def get_vector(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get a vector from the database.
        
        Args:
            idx: Index of the vector
            
        Returns:
            Tuple of (vector, metadata)
        """
        if idx < 0 or idx >= len(self.vectors):
            raise IndexError(f"Vector index out of range: {idx}")

        # Update usage info
        self.usage_info[idx]["last_accessed"] = time.time()
        self.usage_info[idx]["access_count"] += 1

        return self.vectors[idx], self.metadata[idx]

    def update_vector(self, idx: int, vector: np.ndarray = None, metadata: Dict[str, Any] = None) -> bool:
        """Update a vector in the database.
        
        Args:
            idx: Index of the vector
            vector: New vector (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if idx < 0 or idx >= len(self.vectors):
            return False

        # Update vector if provided
        if vector is not None:
            # Check vector dimension
            if vector.shape[0] != self.config["dimension"]:
                raise ValueError(f"Vector dimension mismatch: {vector.shape[0]} != {self.config['dimension']}")

            self.vectors[idx] = vector

        # Update metadata if provided
        if metadata is not None:
            self.metadata[idx].update(metadata)

        # Update usage info
        self.usage_info[idx]["last_accessed"] = time.time()
        self.usage_info[idx]["access_count"] += 1

        # Save changes
        self._save_vectors()

        return True

    def delete_vector(self, idx: int) -> bool:
        """Delete a vector from the database.
        
        Args:
            idx: Index of the vector
            
        Returns:
            True if successful, False otherwise
        """
        if idx < 0 or idx >= len(self.vectors):
            return False

        # Delete vector
        self.vectors = np.delete(self.vectors, idx, axis=0)
        self.metadata.pop(idx)
        self.usage_info.pop(idx)

        # Save changes
        self._save_vectors()

        return True

    def get_all_vectors(self) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """Get all vectors from the database.
        
        Returns:
            List of (index, vector, metadata) tuples
        """
        return [(i, self.vectors[i], self.metadata[i]) for i in range(len(self.vectors))]

    def clear(self) -> bool:
        """Clear the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectors = np.zeros((0, self.config["dimension"]))
            self.metadata = []
            self.usage_info = []

            # Save changes
            self._save_vectors()

            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False


# Initialize vector database with default config
_vector_db = None


def get_vector_db() -> MemoryVectorDB:
    """Get the global vector database instance."""
    global _vector_db

    if _vector_db is None:
        # Try to find config file
        config_path = os.path.join(os.getcwd(), "core", "memory", "vector_db_config.json")
        if not os.path.exists(config_path):
            # Try relative path
            config_path = os.path.join(os.path.dirname(__file__), "vector_db_config.json")

        _vector_db = MemoryVectorDB(config_path)

    return _vector_db
