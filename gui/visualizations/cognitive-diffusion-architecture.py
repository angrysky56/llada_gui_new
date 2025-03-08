#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cognitive Diffusion Framework: Integrating LLaDA with MCP Titan Memory

This framework combines diffusion-based language generation with neural memory
to create a system capable of guided diffusion with memory persistence.
"""

import numpy as np
import requests
import torch
from llada.config import ModelConfig
from llada.masking import LowConfidenceMasker, RandomMasker
# LLaDA components
from llada.model import DiffusionLM
from llada.utils import tensor_util
from transformers import AutoTokenizer


# Memory system interface (communicates with MCP Titan)
class MCPTitanMemoryInterface:
    """Interface to the MCP Titan Memory system via HTTP API."""

    def __init__(self, api_url="http://localhost:3000/api"):
        """Initialize the memory interface.
        
        Args:
            api_url: URL of the MCP Titan Memory API
        """
        self.api_url = api_url
        self.memory_state = None
        self.input_dim = 64  # Default, will be updated from model
        self.memory_dim = 64  # Default, will be updated from model

    def initialize(self, input_dim=64, memory_dim=64):
        """Initialize the memory model.
        
        Args:
            input_dim: Dimension of input vectors
            memory_dim: Dimension of memory vectors
        """
        response = requests.post(
            f"{self.api_url}/init_model",
            json={"inputDim": input_dim, "outputDim": memory_dim}
        )
        response.raise_for_status()

        self.input_dim = input_dim
        self.memory_dim = memory_dim

        # Initialize memory state to zeros
        self.memory_state = np.zeros(memory_dim)

        return response.json()

    def forward_pass(self, input_vector):
        """Run forward pass through the memory model.
        
        Args:
            input_vector: Input vector of shape [input_dim]
            
        Returns:
            dict with predicted, newMemory, and surprise
        """
        if self.memory_state is None:
            raise ValueError("Memory not initialized. Call initialize() first.")

        response = requests.post(
            f"{self.api_url}/forward_pass",
            json={
                "x": input_vector.tolist(),
                "memoryState": self.memory_state.tolist()
            }
        )
        response.raise_for_status()
        result = response.json()

        # Update memory state
        self.memory_state = np.array(result["newMemory"])

        return result

    def train_step(self, current_vector, next_vector):
        """Train the memory on a transition.
        
        Args:
            current_vector: Current state vector
            next_vector: Next state vector
            
        Returns:
            Loss value
        """
        if self.memory_state is None:
            raise ValueError("Memory not initialized. Call initialize() first.")

        response = requests.post(
            f"{self.api_url}/train_step",
            json={
                "x_t": current_vector.tolist(),
                "x_next": next_vector.tolist(),
                "memoryState": self.memory_state.tolist()
            }
        )
        response.raise_for_status()
        result = response.json()

        # Update memory state based on training
        # Use forward_pass to update memory
        self.forward_pass(current_vector)

        return result["loss"]

    def get_memory_state(self):
        """Get the current memory state.
        
        Returns:
            Current memory state vector
        """
        return self.memory_state

    def set_memory_state(self, memory_state):
        """Set the memory state manually.
        
        Args:
            memory_state: New memory state vector
        """
        if len(memory_state) != self.memory_dim:
            raise ValueError(f"Memory state must have dimension {self.memory_dim}")

        self.memory_state = np.array(memory_state)


class CognitiveDiffusionSystem:
    """Core system that integrates diffusion LM with memory guidance."""

    def __init__(self, llada_model_path, device="cuda"):
        """Initialize the cognitive diffusion system.
        
        Args:
            llada_model_path: Path to LLaDA model
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Load LLaDA model
        self.config = ModelConfig()
        self.model = DiffusionLM.from_pretrained(llada_model_path, config=self.config)
        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llada_model_path)

        # Initialize memory system
        self.memory = MCPTitanMemoryInterface()

        # Determine embedding dimensions from tokenizer
        vocab_size = len(self.tokenizer)
        self.token_embedding_dim = 64  # This should match what MCP Titan expects

        # Initialize memory with appropriate dimensions
        self.memory.initialize(input_dim=self.token_embedding_dim, memory_dim=self.token_embedding_dim)

        # Create token embedding lookup
        self.token_embeddings = self._initialize_token_embeddings(vocab_size, self.token_embedding_dim)

    @staticmethod
    def _initialize_token_embeddings(vocab_size, embedding_dim):
        """Initialize token embeddings for memory interface.
        
        This creates a mapping from token IDs to vector representations
        that the memory system can work with.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            
        Returns:
            Tensor of token embeddings
        """
        # For now, we use random embeddings that will get tuned
        # In a full implementation, these could be extracted from the LLaDA model
        # or from a separate embedding model
        embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        return embeddings

    def _sequence_to_embeddings(self, token_ids):
        """Convert a sequence of token IDs to embeddings.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            Tensor of token embeddings
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=self.device)
        return self.token_embeddings(token_ids)

    def _update_memory_with_sequence(self, token_ids):
        """Train memory on a sequence of tokens.
        
        Args:
            token_ids: List or tensor of token IDs
        """
        if len(token_ids) < 2:
            return  # Need at least 2 tokens to create a transition

        # Convert tokens to embeddings
        embeddings = self._sequence_to_embeddings(token_ids)

        # Train memory on transitions
        for i in range(len(embeddings) - 1):
            current_emb = embeddings[i].cpu().detach().numpy()
            next_emb = embeddings[i + 1].cpu().detach().numpy()
            self.memory.train_step(current_emb, next_emb)

    def generate(self, prompt, gen_length=64, steps=64, block_length=32,
                 temperature=1.0, cfg_scale=0.0,
                 memory_weight=0.3, remasking="low_confidence"):
        """Generate text using memory-guided diffusion.
        
        Args:
            prompt: Input prompt
            gen_length: Length of generation
            steps: Number of diffusion steps
            block_length: Block length for masking
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale
            memory_weight: Weight of memory guidance (0-1)
            remasking: Masking strategy ("low_confidence" or "random")
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = tokens.shape[1]

        # Pad to target length
        if input_length < gen_length:
            # For real implementation, use proper padding
            padded = torch.zeros((1, gen_length), dtype=torch.long, device=self.device)
            padded[0, :input_length] = tokens[0]
            tokens = padded

        # Initialize masking strategy
        if remasking == "low_confidence":
            masker = LowConfidenceMasker(block_size=block_length)
        else:
            masker = RandomMasker(block_size=block_length)

        # Initial masks: mask everything except the prompt
        masks = torch.ones((1, gen_length), dtype=torch.bool, device=self.device)
        masks[0, :input_length] = False  # Unmask prompt tokens

        # Set up visualization tracking
        all_steps = []

        # Main diffusion loop
        for step in range(steps):
            # Extract current token IDs
            current_tokens = tokens[0, :].cpu().detach().tolist()

            # Perform LLaDA diffusion step to get token probabilities
            with torch.no_grad():
                outputs = self.model(
                    tokens,
                    masks=masks,
                    temperature=temperature,
                    cfg_scale=cfg_scale
                )

            # Get token logits and confidences
            logits = outputs["logits"]
            confidences = outputs["confidences"]

            # Convert to probabilities
            token_probs = torch.softmax(logits / temperature, dim=-1)

            # Memory guidance for tokens that are still masked
            for pos in range(gen_length):
                if masks[0, pos].item():  # If position is masked
                    # Get previous token embedding for context
                    prev_pos = max(0, pos - 1)
                    prev_token_id = tokens[0, prev_pos].item()
                    prev_embedding = self._sequence_to_embeddings([prev_token_id])[0].cpu().detach().numpy()

                    # Query memory for prediction
                    memory_result = self.memory.forward_pass(prev_embedding)
                    predicted_vector = np.array(memory_result["predicted"])
                    surprise_value = memory_result["surprise"]

                    # Find token most similar to predicted vector
                    all_embeddings = self.token_embeddings.weight.cpu().detach().numpy()
                    similarities = np.dot(all_embeddings, predicted_vector) / (
                            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(predicted_vector) + 1e-8
                    )

                    # Create memory-based distribution
                    memory_probs = torch.zeros_like(token_probs[0, pos])

                    # Set probabilities based on similarity
                    for token_id, sim in enumerate(similarities):
                        # Scale to [0, 1] and add small epsilon
                        sim_prob = (sim + 1) / 2 + 1e-8
                        memory_probs[token_id] = sim_prob

                    # Normalize memory probabilities
                    memory_probs = memory_probs / memory_probs.sum()

                    # Combine with model probabilities (weighted sum)
                    combined_probs = (1 - memory_weight) * token_probs[0, pos] + memory_weight * memory_probs

                    # Update token probabilities
                    token_probs[0, pos] = combined_probs

            # Sample new tokens for masked positions
            for pos in range(gen_length):
                if masks[0, pos].item():  # If position is masked
                    # Sample from combined distribution
                    token_id = torch.multinomial(token_probs[0, pos], 1).item()
                    tokens[0, pos] = token_id

            # Update masks with remasking strategy
            masks = masker.update_masks(masks, confidences)

            # Store step data for visualization
            all_steps.append({
                "step": step,
                "tokens": tokens[0, :].cpu().tolist(),
                "masks": masks[0, :].cpu().tolist(),
                "confidences": confidences[0, :].cpu().tolist()
            })

            # Check if we're done (no masks left)
            if not masks.any():
                break

            # Update memory with current token sequence
            # This trains the memory on what's been generated so far
            self._update_memory_with_sequence(tokens[0, :input_length + step + 1].cpu().tolist())

        # Decode the final tokens
        output_text = self.tokenizer.decode(tokens[0, :], skip_special_tokens=True)

        return {
            "text": output_text,
            "steps": all_steps
        }

    def visualize_memory_influence(self, text):
        """Visualize the influence of memory on text generation.
        
        This method shows how different parts of the text were influenced
        by the memory system vs. the base model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Visualization data
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        embeddings = self._sequence_to_embeddings(tokens[0, :])

        # Analyze each token's relationship to memory
        token_influences = []

        prev_memory_state = self.memory.get_memory_state()

        for i in range(len(tokens[0])):
            token_id = tokens[0, i].item()
            token_text = self.tokenizer.decode([token_id])
            token_embedding = embeddings[i].cpu().detach().numpy()

            # Get memory prediction from previous state
            if i > 0:
                prev_token_embedding = embeddings[i - 1].cpu().detach().numpy()
                memory_result = self.memory.forward_pass(prev_token_embedding)
                predicted_vector = np.array(memory_result["predicted"])
                surprise_value = memory_result["surprise"]

                # Calculate cosine similarity between predicted and actual
                cos_sim = np.dot(predicted_vector, token_embedding) / (
                        np.linalg.norm(predicted_vector) * np.linalg.norm(token_embedding) + 1e-8
                )

                # Reset memory state to continue analysis
                self.memory.set_memory_state(prev_memory_state)
            else:
                cos_sim = 0.0
                surprise_value = 0.0

            token_influences.append({
                "token": token_text,
                "token_id": token_id,
                "memory_similarity": float(cos_sim),
                "surprise": float(surprise_value)
            })

        return {
            "text": text,
            "token_influences": token_influences
        }
