#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model caching system for LLaDA.

This module provides a caching system for the LLaDA model to avoid reloading
the model from disk on each generation, significantly improving startup time
for subsequent generations.
"""

import gc
import hashlib
import logging
import threading
import time

import torch

# Configure logging
logger = logging.getLogger(__name__)

# Singleton model cache
_model_cache = None


def get_model_cache():
    """
    Get or create the model cache instance.
    
    Returns:
        ModelCache: The singleton model cache instance
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


class ModelCache:
    """
    Cache for the LLaDA model to avoid repeated loading.
    
    This class provides a memory cache for the model to avoid loading it from disk
    on each generation request, significantly improving startup time after the first run.
    
    It can cache models with different configurations (e.g., 4-bit, 8-bit) and
    automatically manages memory usage.
    """

    def __init__(self, max_cache_size=2):
        """
        Initialize the model cache.
        
        Args:
            max_cache_size: Maximum number of model variants to cache
        """
        self.lock = threading.RLock()
        self.cache = {}  # Config hash -> (model, tokenizer, timestamp)
        self.max_cache_size = max_cache_size
        logger.info(f"Model cache initialized with max size {max_cache_size}")

    def get(self, config):
        """
        Get a model from the cache if available.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            tuple: (model, tokenizer) if cached, (None, None) otherwise
        """
        with self.lock:
            # Create a hash of the relevant configuration
            config_hash = self._hash_config(config)

            # Check if the model is in the cache
            if config_hash in self.cache:
                model, tokenizer, timestamp = self.cache[config_hash]
                # Update the timestamp
                self.cache[config_hash] = (model, tokenizer, time.time())
                logger.info(f"Model found in cache for config {config_hash}")
                return model, tokenizer

            logger.info(f"Model not found in cache for config {config_hash}")
            return None, None

    def put(self, config, model, tokenizer):
        """
        Add a model to the cache.
        
        Args:
            config: Model configuration dictionary
            model: The model to cache
            tokenizer: The tokenizer to cache
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        with self.lock:
            # Create a hash of the relevant configuration
            config_hash = self._hash_config(config)

            # If already in cache, just update it
            if config_hash in self.cache:
                self.cache[config_hash] = (model, tokenizer, time.time())
                logger.info(f"Updated existing model in cache for config {config_hash}")
                return True

            # Check if we need to remove an old entry
            if len(self.cache) >= self.max_cache_size:
                self._evict_oldest()

            # Add to cache
            self.cache[config_hash] = (model, tokenizer, time.time())
            logger.info(f"Added model to cache for config {config_hash}")
            return True

    def clear(self):
        """
        Clear the entire cache.
        
        Returns:
            bool: True if cleared successfully
        """
        with self.lock:
            # Clear references
            self.cache.clear()

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model cache cleared")
            return True

    def _evict_oldest(self):
        """
        Evict the oldest entry from the cache.
        
        Returns:
            bool: True if evicted successfully, False otherwise
        """
        with self.lock:
            if not self.cache:
                return False

            # Find the oldest entry
            oldest_hash = None
            oldest_time = float('inf')

            for config_hash, (_, _, timestamp) in self.cache.items():
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_hash = config_hash

            # Remove the oldest entry
            if oldest_hash:
                del self.cache[oldest_hash]

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"Evicted oldest model from cache: {oldest_hash}")
                return True

            return False

    @staticmethod
    def _hash_config(config):
        """
        Create a hash string from the relevant configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            str: Hash string of the configuration
        """
        # Extract relevant keys for caching
        cache_keys = {
            'device': config.get('device', 'cuda'),
            'use_8bit': config.get('use_8bit', False),
            'use_4bit': config.get('use_4bit', False),
            'extreme_mode': config.get('extreme_mode', False)
        }

        # Create a string representation of the config
        config_str = str(sorted(cache_keys.items()))

        # Create a hash
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_stats(self):
        """
        Get statistics about the cache.
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.max_cache_size,
                'cached_configs': [k for k in self.cache.keys()]
            }
