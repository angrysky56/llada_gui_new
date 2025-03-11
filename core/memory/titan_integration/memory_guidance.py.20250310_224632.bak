#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory guidance for LLaDA diffusion models using the Titan Memory system.

This module provides guidance to the diffusion sampling process
based on memory context from previous generations.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

# Import Titan Memory
from ..titan_memory import TitanMemorySystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TitanMemoryGuidance:
    """
    Guidance system for LLaDA diffusion models using Titan Memory.
    
    This class provides:
    1. Integration with the diffusion sampling process
    2. Memory-based token probability adjustment
    3. State tracking and updates based on token sequences
    """

    def __init__(self,
                 memory_system: Optional[TitanMemorySystem] = None,
                 memory_weight: float = 0.3,
                 tokenizer=None):
        """Initialize memory guidance.
        
        Args:
            memory_system: TitanMemorySystem instance or None
            memory_weight: Influence weight for memory guidance (0-1)
            tokenizer: Optional tokenizer for better text handling
        """
        self.memory_system = memory_system or TitanMemorySystem()
        self.memory_weight = memory_weight
        self.tokenizer = tokenizer
        self.initialized = self.memory_system.initialized
        self.token_embeddings = None
        self.current_state = None
        self.mask_id = 126336  # Default mask ID, will be updated if tokenizer provides one
        self.confidence_threshold = 0.7  # Minimum confidence to use memory
        self.auto_train = True  # Enable automatic training by default

        # Internal tracking
        self.generation_history = []
        self.embedding_cache = {}

        # Try to initialize a simple token embedding if none is provided
        if tokenizer:
            # Try to get vocabulary size
            if hasattr(tokenizer, 'vocab_size'):
                self._init_simple_embeddings(tokenizer.vocab_size)
            elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
                self._init_simple_embeddings(len(tokenizer.vocab))

    def _init_simple_embeddings(self, vocab_size):
        """Initialize simple embeddings for tokens.
        
        This creates a random but consistent embedding space for tokens,
        which can be used when the model doesn't provide embeddings.
        
        Args:
            vocab_size: Size of the vocabulary
        """
        # Use a fixed random seed for consistency
        rng = np.random.RandomState(42)

        # Create simple embeddings - normalize for better performance
        embeddings = rng.randn(vocab_size, 64)
        embedding_norms = np.sqrt((embeddings ** 2).sum(axis=1, keepdims=True))
        normalized_embeddings = embeddings / (embedding_norms + 1e-8)

        self.token_embeddings = normalized_embeddings
        logger.info(f"Initialized simple embeddings with shape {normalized_embeddings.shape}")

    def get_token_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding for a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Embedding vector
        """
        # Check if we have the token in cache first
        if token_id in self.embedding_cache:
            return self.embedding_cache[token_id]

        # Use token embeddings if available
        if self.token_embeddings is not None and token_id < len(self.token_embeddings):
            embedding = self.token_embeddings[token_id]
            self.embedding_cache[token_id] = embedding
            return embedding

        # Fall back to a hash-based embedding
        # This creates a deterministic but unique embedding for any token ID
        embedding = np.zeros(64)
        embedding_seed = (token_id * 1664525 + 1013904223) % (2 ** 32)  # Simple LCG hash
        rng = np.random.RandomState(embedding_seed)
        embedding = rng.randn(64)

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache for future use
        self.embedding_cache[token_id] = embedding
        return embedding

    def set_memory_weight(self, weight: float):
        """Set the memory guidance weight.
        
        Args:
            weight: Influence weight (0-1)
        """
        self.memory_weight = max(0.0, min(1.0, weight))

    def get_memory_state(self) -> np.ndarray:
        """Get the current memory state.
        
        Returns:
            Memory state as numpy array
        """
        if self.initialized and self.memory_system:
            return np.array(self.memory_system.get_memory_state())
        return np.zeros(64)

    def is_initialized(self) -> bool:
        """Check if memory system is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self.initialized and self.memory_system.initialized

    def reset(self):
        """Reset the memory state."""
        if self.initialized and self.memory_system:
            self.memory_system.reset_memory()

    def update_memory_from_tokens(self, token_sequence: List[int]) -> Tuple[np.ndarray, float]:
        """Update memory state based on token sequence.
        
        Args:
            token_sequence: List of token IDs
            
        Returns:
            Tuple of (new memory state, surprise)
        """
        if not self.initialized or not self.memory_system:
            return np.zeros(64), 0.0

        # Create a simple embedding of the token sequence
        # This is just a simple aggregation for demonstration
        embedding = np.zeros(64)

        for token_id in token_sequence[-100:]:  # Use only the last 100 tokens
            # Skip mask tokens
            if self.mask_id is not None and token_id == self.mask_id:
                continue

            # Get token embedding
            token_embedding = self.get_token_embedding(token_id)

            # Add to sequence embedding with position-based weighting
            # More recent tokens have higher weights
            position_weight = len(embedding) / 100.0  # Simple position weight scaling
            embedding += token_embedding * position_weight

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Update memory with the embedding
        new_memory, surprise = self.memory_system.update_memory(embedding)

        return np.array(new_memory), surprise

    def apply_memory_guidance(self,
                              token_probs: np.ndarray,
                              token_sequence: List[int],
                              vocab_size: int) -> np.ndarray:
        """Apply memory guidance to token probabilities.
        
        Args:
            token_probs: Original token probabilities
            token_sequence: Current token sequence
            vocab_size: Size of the vocabulary
            
        Returns:
            Updated token probabilities
        """
        if not self.initialized or not self.memory_system or self.memory_weight <= 0.0:
            return token_probs

        try:
            # Create a sequence embedding
            seq_embedding = np.zeros(64)

            # Create embedding from the most recent tokens
            for token_id in token_sequence[-100:]:  # Use last 100 tokens
                # Skip mask tokens
                if self.mask_id is not None and token_id == self.mask_id:
                    continue

                token_embedding = self.get_token_embedding(token_id)
                seq_embedding += token_embedding

            # Normalize the embedding
            norm = np.linalg.norm(seq_embedding)
            if norm > 0:
                seq_embedding = seq_embedding / norm

            # Get prediction from memory system
            result = self.memory_system.forward_pass(seq_embedding)

            # Check if we have a valid prediction
            if "predicted" not in result:
                return token_probs

            # Get the predicted token embedding
            predicted_embedding = np.array(result["predicted"])

            # Calculate similarity scores with all token embeddings
            memory_probs = np.zeros_like(token_probs)
            for token_id in range(
                    min(vocab_size, len(self.token_embeddings) if self.token_embeddings is not None else 0)):
                token_embedding = self.get_token_embedding(token_id)
                # Calculate cosine similarity
                similarity = np.dot(predicted_embedding, token_embedding)
                # Convert to probability (scaled to [0, 1])
                memory_probs[token_id] = (similarity + 1) / 2.0

            # If token embeddings don't cover the full vocabulary, assign uniform probabilities to the rest
            if self.token_embeddings is not None and vocab_size > len(self.token_embeddings):
                memory_probs[len(self.token_embeddings):] = 0.5  # Default probability for unknown tokens

            # Normalize memory probabilities
            memory_probs_sum = memory_probs.sum()
            if memory_probs_sum > 0:
                memory_probs = memory_probs / memory_probs_sum
            else:
                # Fall back to uniform distribution - with safe casting
                try:
                    memory_probs = np.ones_like(memory_probs, dtype=np.float64) / len(memory_probs)
                except Exception as e:
                    logger.warning(f"Casting error in memory guidance: {e}")
                    # Ultra-safe fallback
                    memory_probs = np.ones(len(memory_probs), dtype=np.float64) / len(memory_probs)

            # Blend with original probabilities based on memory weight
            surprise = float(result.get("surprise", 0.0))

            # Adaptive weighting based on surprise
            # High surprise = low confidence, so reduce memory weight
            confidence = 1.0 / (1.0 + surprise)
            adaptive_weight = self.memory_weight * min(confidence, self.confidence_threshold)

            # Blend probabilities
            guided_probs = (1 - adaptive_weight) * token_probs + adaptive_weight * memory_probs

            # Ensure proper normalization
            guided_probs_sum = guided_probs.sum()
            if guided_probs_sum > 0:
                guided_probs = guided_probs / guided_probs_sum

            return guided_probs

        except Exception as e:
            logger.error(f"Error in memory guidance: {e}")
            return token_probs

    def train_on_generation(self, prompt: str, generation: str) -> float:
        """Train memory on a prompt-generation pair.
        
        Args:
            prompt: Input prompt
            generation: Generated text
            
        Returns:
            Training loss
        """
        if not self.initialized or not self.memory_system:
            return 0.0

        try:
            # Create simple embeddings
            def create_text_embedding(text):
                embedding = np.zeros(64)
                for i, char in enumerate(text[:1000]):
                    embedding[i % 64] += ord(char) % 10

                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                return embedding

            # Create embeddings for prompt and generation
            prompt_embedding = create_text_embedding(prompt)
            generation_embedding = create_text_embedding(generation)

            # Train the memory system
            loss = self.memory_system.train_step(prompt_embedding, generation_embedding)

            return loss
        except Exception as e:
            logger.error(f"Error training memory: {e}")
            return 0.0

    def save_model(self, path: Optional[str] = None):
        """Save the memory model.
        
        Args:
            path: Optional save path
        """
        if self.initialized and self.memory_system:
            try:
                self.memory_system.save_model(path)
                return True
            except Exception as e:
                logger.error(f"Error saving memory model: {e}")
        return False

    def load_model(self, path: Optional[str] = None):
        """Load the memory model.
        
        Args:
            path: Optional load path
        """
        if self.initialized and self.memory_system:
            try:
                self.memory_system.load_model(path)
                return True
            except Exception as e:
                logger.error(f"Error loading memory model: {e}")
        return False
