#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Embeddings System for LLaDA GUI

This module provides functionality to convert between tokenized text and vector 
embeddings that can be used with the Titan Memory system. It creates a bridge
between the text domain and the vector domain needed for memory-guided diffusion.
"""

import os
import sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any


class TokenEmbeddings:
    """
    Manages embeddings for tokens in the LLaDA vocabulary.
    
    This class provides conversion between token IDs and vector representations
    that can be used with the Titan Memory system.
    """
    
    def __init__(self, 
                vocab_size: int, 
                embedding_dim: int = 64, 
                pretrained_path: Optional[str] = None):
        """Initialize the token embeddings.
        
        Args:
            vocab_size: Size of the token vocabulary
            embedding_dim: Dimension of embedding vectors
            pretrained_path: Optional path to pretrained embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load pretrained embeddings if available
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_embeddings(pretrained_path)
        else:
            # Initialize with normalized random vectors
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            self._normalize_embeddings()
    
    def _normalize_embeddings(self):
        """Normalize embedding vectors to unit length."""
        with torch.no_grad():
            norm = torch.norm(self.embedding.weight, dim=1, keepdim=True)
            self.embedding.weight.data = self.embedding.weight.data / (norm + 1e-8)
    
    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding vector for a token ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Embedding vector as numpy array
        """
        with torch.no_grad():
            if token_id >= self.vocab_size:
                # Handle out of vocabulary token
                token_id = token_id % self.vocab_size
                
            vector = self.embedding(torch.tensor(token_id)).numpy()
            return vector
    
    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Get embedding vectors for multiple token IDs.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Array of embedding vectors
        """
        with torch.no_grad():
            # Handle out of vocabulary tokens
            token_ids = [t % self.vocab_size for t in token_ids]
            vectors = self.embedding(torch.tensor(token_ids)).numpy()
            return vectors
    
    def get_nearest_tokens(self, 
                          vector: Union[np.ndarray, List[float]], 
                          k: int = 5) -> List[Tuple[int, float]]:
        """Find the nearest tokens to a given vector.
        
        Args:
            vector: Query vector
            k: Number of neighbors to return
            
        Returns:
            List of (token_id, similarity) tuples
        """
        # Convert vector to tensor
        if isinstance(vector, np.ndarray):
            query = torch.from_numpy(vector).float()
        elif isinstance(vector, list):
            query = torch.tensor(vector, dtype=torch.float32)
        else:
            query = vector
        
        # Normalize query
        query = F.normalize(query, dim=0)
        
        # Compute similarities
        with torch.no_grad():
            similarities = F.cosine_similarity(
                query.unsqueeze(0), 
                self.embedding.weight,
                dim=1
            )
        
        # Get top k
        values, indices = torch.topk(similarities, k=min(k, self.vocab_size))
        
        return list(zip(indices.tolist(), values.tolist()))
    
    def save_embeddings(self, path: str):
        """Save embeddings to file.
        
        Args:
            path: Path to save file
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save embeddings
        with torch.no_grad():
            embeddings = self.embedding.weight.cpu().numpy()
            np.save(path, embeddings)
    
    def load_embeddings(self, path: str):
        """Load embeddings from file.
        
        Args:
            path: Path to embeddings file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        # Load embeddings
        embeddings = np.load(path)
        
        # Check dimensions
        if embeddings.shape[0] != self.vocab_size or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimensions mismatch. Expected {self.vocab_size}x{self.embedding_dim}, "
                f"got {embeddings.shape[0]}x{embeddings.shape[1]}"
            )
        
        # Update embedding weights
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(embeddings))


class ContextEncoder:
    """
    Encodes context from token sequences for the memory system.
    
    This class handles converting token sequences to context vectors
    that encapsulate the semantic meaning for memory storage.
    """
    
    def __init__(self, 
                token_embeddings: TokenEmbeddings,
                context_size: int = 3,
                output_dim: int = 64):
        """Initialize the context encoder.
        
        Args:
            token_embeddings: Token embeddings object
            context_size: Number of tokens to use for context
            output_dim: Dimension of output vectors
        """
        self.token_embeddings = token_embeddings
        self.context_size = context_size
        self.output_dim = output_dim
        
        # Add a simple context adaptation layer
        self.adapter = nn.Linear(
            token_embeddings.embedding_dim * context_size,
            output_dim
        )
    
    def encode_context(self, 
                      token_ids: List[int], 
                      position: int) -> np.ndarray:
        """Encode context at a specific position.
        
        Args:
            token_ids: List of token IDs
            position: Position to encode context for
            
        Returns:
            Context vector
        """
        # Get context window
        start = max(0, position - self.context_size + 1)
        context_tokens = token_ids[start:position+1]
        
        # Pad if needed
        if len(context_tokens) < self.context_size:
            context_tokens = [0] * (self.context_size - len(context_tokens)) + context_tokens
        
        # Get embeddings
        embeddings = self.token_embeddings.get_embeddings(context_tokens)
        
        # Flatten embeddings
        flat = embeddings.reshape(-1)
        
        # Apply adapter
        with torch.no_grad():
            tensor = torch.from_numpy(flat).float()
            output = self.adapter(tensor)
            
        return output.numpy()
    
    def encode_sequence(self, token_ids: List[int]) -> List[np.ndarray]:
        """Encode all positions in a sequence.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of context vectors
        """
        return [
            self.encode_context(token_ids, i)
            for i in range(len(token_ids))
        ]


class EmbeddingAugmentedMemory:
    """
    Bridge between token sequences and memory vectors.
    
    This class provides a high-level interface for using token embeddings
    with the Titan Memory system.
    """
    
    def __init__(self, 
                vocab_size: int, 
                embedding_dim: int = 64,
                memory_dim: int = 64,
                context_size: int = 3,
                embeddings_path: Optional[str] = None):
        """Initialize the embedding-augmented memory.
        
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            memory_dim: Dimension of memory vectors
            context_size: Context window size
            embeddings_path: Optional path to pretrained embeddings
        """
        # Create token embeddings
        self.token_embeddings = TokenEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_path=embeddings_path
        )
        
        # Create context encoder
        self.context_encoder = ContextEncoder(
            token_embeddings=self.token_embeddings,
            context_size=context_size,
            output_dim=memory_dim
        )
        
        # Create data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "memory_data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Default embeddings path
        self.default_embeddings_path = os.path.join(
            self.data_dir, 
            "token_embeddings.npy"
        )
        
        # Try to save embeddings
        try:
            self.token_embeddings.save_embeddings(self.default_embeddings_path)
        except Exception as e:
            print(f"Warning: Failed to save embeddings: {str(e)}")
    
    def encode_tokens(self, token_ids: List[int], position: int) -> np.ndarray:
        """Encode tokens at a specific position for memory.
        
        Args:
            token_ids: List of token IDs
            position: Position to encode
            
        Returns:
            Memory-compatible vector
        """
        return self.context_encoder.encode_context(token_ids, position)
    
    def predict_next_token(self, 
                          token_ids: List[int], 
                          memory_prediction: np.ndarray, 
                          k: int = 10) -> List[Tuple[int, float]]:
        """Predict the next token based on memory prediction.
        
        Args:
            token_ids: Previous token IDs
            memory_prediction: Memory system's prediction vector
            k: Number of candidates to return
            
        Returns:
            List of (token_id, probability) tuples
        """
        # Find nearest tokens to the prediction
        nearest = self.token_embeddings.get_nearest_tokens(memory_prediction, k=k)
        
        # Normalize scores to probabilities
        total = sum(max(0, s) for _, s in nearest)
        if total > 0:
            nearest = [(t, max(0, s) / total) for t, s in nearest]
        else:
            # If all similarities are negative, use softmax
            scores = np.array([s for _, s in nearest])
            probs = F.softmax(torch.from_numpy(scores), dim=0).numpy()
            nearest = [(t, p) for (t, _), p in zip(nearest, probs)]
        
        return nearest
    
    def train_on_sequence(self, token_ids: List[int], memory_system) -> List[float]:
        """Train the memory system on a token sequence.
        
        Args:
            token_ids: List of token IDs
            memory_system: Titan Memory System instance
            
        Returns:
            List of training loss values
        """
        if len(token_ids) < 2:
            return []
        
        # Encode all positions
        encoded = self.context_encoder.encode_sequence(token_ids)
        
        # Train on pairs of consecutive vectors
        losses = []
        for i in range(len(encoded) - 1):
            loss = memory_system.train_step(encoded[i], encoded[i+1])
            losses.append(loss)
        
        # Save embeddings after training
        try:
            self.token_embeddings.save_embeddings(self.default_embeddings_path)
        except Exception as e:
            print(f"Warning: Failed to save embeddings after training: {str(e)}")
        
        return losses


# Simple test
def test_embeddings():
    """Test the embedding system."""
    print("Testing Embedding System...")
    
    # Create embedding system
    augmented_memory = EmbeddingAugmentedMemory(
        vocab_size=10000,  # Example vocab size
        embedding_dim=64,
        memory_dim=64,
        context_size=3
    )
    
    # Test token encoding
    token_ids = [1, 2, 3, 4, 5]
    encoded = augmented_memory.encode_tokens(token_ids, 3)
    print(f"Encoded vector shape: {encoded.shape}")
    
    # Test token prediction
    memory_prediction = np.random.randn(64)
    predicted = augmented_memory.predict_next_token(token_ids, memory_prediction, k=5)
    print(f"Predicted tokens: {predicted}")
    
    print("Test complete!")
    return True


if __name__ == "__main__":
    test_embeddings()
