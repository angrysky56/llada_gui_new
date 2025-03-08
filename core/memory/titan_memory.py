#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Titan Memory System - Integrated Version for LLaDA GUI

A modern implementation of the MCP Titan Memory system, rebuilt to work directly
within the LLaDA GUI without external dependencies or deprecated packages.

Based on the concepts from Google Research's paper 
"Generative AI for Programming: A Common Task Framework"
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class TitanMemoryConfig:
    """Configuration for the Titan Memory model."""
    input_dim: int = 64  # Input dimension
    hidden_dim: int = 32  # Hidden layer dimension
    memory_dim: int = 64  # Memory state dimension
    learning_rate: float = 1e-3  # Learning rate for optimizer
    use_manifold: bool = False  # Use manifold optimization
    momentum_factor: float = 0.9  # Momentum for optimizer
    forget_gate_init: float = 0.01  # Initial forget gate value
    max_step_size: float = 0.1  # Maximum step size for manifold updates
    tangent_epsilon: float = 1e-8  # Small epsilon for numerical stability


class TitanMemoryModel(nn.Module):
    """
    Neural memory model based on the Titan Memory system.
    
    This model maintains a memory state and can predict the next token based
    on the current input and memory state. It uses a simple MLP architecture
    with a forget gate mechanism.
    """

    def __init__(self, config: TitanMemoryConfig = None):
        """Initialize the Titan Memory Model.
        
        Args:
            config: Configuration parameters for the model
        """
        super(TitanMemoryModel, self).__init__()

        # Use default config if none provided
        if config is None:
            config = TitanMemoryConfig()

        # Store configuration
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.memory_dim = config.memory_dim
        self.full_output_dim = self.input_dim + self.memory_dim

        # Initialize layers
        # First layer: input + memory -> hidden
        self.fc1 = nn.Linear(self.input_dim + self.memory_dim, self.hidden_dim)

        # Second layer: hidden -> output (input prediction + new memory)
        self.fc2 = nn.Linear(self.hidden_dim, self.full_output_dim)

        # Forget gate (learnable scalar parameter)
        self.forget_gate = nn.Parameter(torch.tensor(config.forget_gate_init))

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)

        # Initialize memory state
        self.reset_memory()

    def reset_memory(self):
        """Reset the memory state to zeros."""
        self.memory_state = torch.zeros(self.memory_dim)

    def get_memory_state(self) -> np.ndarray:
        """Get the current memory state.
        
        Returns:
            Current memory state as numpy array
        """
        return self.memory_state.detach().cpu().numpy()

    def set_memory_state(self, state: Union[np.ndarray, torch.Tensor, List[float]]):
        """Set the memory state manually.
        
        Args:
            state: New memory state
        """
        if isinstance(state, np.ndarray):
            self.memory_state = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            self.memory_state = state.float()
        elif isinstance(state, list):
            self.memory_state = torch.tensor(state, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported memory state type: {type(state)}")

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [input_dim]
            memory: Optional memory state. If None, use current state.
            
        Returns:
            Dictionary with predicted, new_memory, and surprise values
        """
        if memory is None:
            memory = self.memory_state

        # Apply forget gate to memory state
        forget_value = torch.sigmoid(self.forget_gate)  # Sigmoid to keep it in [0, 1]
        gated_memory = memory * (1 - forget_value)

        # Combine input and gated memory
        combined = torch.cat([x, gated_memory], dim=0)

        # MLP forward pass
        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)

        # Split output into new memory and predicted next input
        new_memory = output[:self.memory_dim]
        predicted = output[self.memory_dim:]

        # Calculate surprise (MSE between predicted and actual input)
        diff = predicted - x
        surprise = torch.mean(diff * diff)

        return {
            "predicted": predicted,
            "new_memory": new_memory,
            "surprise": surprise
        }

    def update_memory(self, x: torch.Tensor):
        """Update memory state based on input.
        
        Args:
            x: Input tensor
            
        Returns:
            New memory state and surprise value
        """
        with torch.no_grad():
            result = self.forward(x)
            new_memory = result["new_memory"].detach().clone()
            surprise = result["surprise"].item()

            # Update memory state
            self.memory_state = new_memory

            return new_memory, surprise

    def train_step(self, x_t: torch.Tensor, x_next: torch.Tensor) -> float:
        """Perform a training step.
        
        Args:
            x_t: Current input tensor
            x_next: Next input tensor (target)
            
        Returns:
            Loss value
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        result = self.forward(x_t)
        predicted = result["predicted"]
        new_memory = result["new_memory"]
        surprise = result["surprise"]

        # Calculate loss (MSE between predicted and next + small surprise penalty)
        diff = predicted - x_next
        mse_loss = torch.mean(diff * diff)
        total_loss = mse_loss + 0.01 * surprise

        # Backward pass and optimize
        total_loss.backward()
        self.optimizer.step()

        # Update memory state
        with torch.no_grad():
            self.memory_state = new_memory

        return total_loss.item()

    def manifold_update(self, base: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Update parameters on the manifold (if enabled).
        
        This implements Riemannian optimization on the unit sphere,
        which can improve stability for embedding vectors.
        
        Args:
            base: Base point on the manifold
            velocity: Update direction
            
        Returns:
            Updated point on the manifold
        """
        if not self.config.use_manifold:
            # Standard Euclidean update
            return base + velocity

        # Riemannian update on the unit sphere
        dot = torch.sum(base * velocity)
        radial = base * dot
        tangent = velocity - radial
        t_norm = torch.norm(tangent)

        # If tangent component is too small, no movement
        if t_norm < self.config.tangent_epsilon:
            return base

        # Limit step size
        step_size = min(t_norm.item(), self.config.max_step_size)
        direction = tangent / t_norm

        # Move along geodesic
        cos_v = torch.cos(torch.tensor(step_size))
        sin_v = torch.sin(torch.tensor(step_size))
        new_point = base * cos_v + direction * sin_v

        # Normalize to stay on the sphere
        return new_point / (torch.norm(new_point) + 1e-12)

    def save_model(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save model state and config
        state_dict = self.state_dict()
        config_dict = vars(self.config)

        # Properly detach tensors and convert to numpy arrays
        processed_state_dict = {}
        for k, v in state_dict.items():
            if k == "forget_gate":
                # Always save forget_gate as a float to ensure compatibility
                processed_state_dict[k] = float(v.item() if isinstance(v, torch.Tensor) else v)
            elif isinstance(v, torch.Tensor):
                processed_state_dict[k] = v.detach().cpu().numpy().tolist()
            else:
                processed_state_dict[k] = v

        save_data = {
            "state_dict": processed_state_dict,
            "config": config_dict,
            "memory_state": self.memory_state.detach().cpu().numpy().tolist()
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

    def load_model(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to model file
        """
        try:
            with open(path, 'r') as f:
                save_data = json.load(f)

            # Load config
            config_dict = save_data.get("config", {})
            self.config = TitanMemoryConfig(**config_dict)

            # Get current state dict as reference for shapes
            current_state_dict = self.state_dict()

            # Initialize tensor parameters
            state_dict = {}
            for k, v in save_data["state_dict"].items():
                # Skip parameters that don't exist in the current model
                if k not in current_state_dict:
                    print(f"Warning: Saved parameter '{k}' not found in current model")
                    continue

                # Handle special case for forget_gate which might be stored as a float
                if k == "forget_gate" and not isinstance(v, list):
                    state_dict[k] = torch.tensor(float(v))
                elif isinstance(v, list):
                    # Check if the dimensions match
                    current_shape = current_state_dict[k].shape
                    saved_shape = torch.tensor(v).shape

                    if saved_shape != current_shape:
                        print(f"Warning: Shape mismatch for {k}: saved {saved_shape}, current {current_shape}")
                        continue

                    state_dict[k] = torch.tensor(v)
                else:
                    # Other parameters (constants, etc.)
                    state_dict[k] = v

            # Only load the state dict if it's not empty
            if state_dict:
                # Load partial state dict (only the keys that exist and have correct shapes)
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded partial state dictionary from {path}")
            else:
                print(f"Warning: No compatible parameters found in saved model")

            # Load memory state
            memory_state = save_data.get("memory_state", None)
            if memory_state:
                self.memory_state = torch.tensor(memory_state)
            else:
                self.reset_memory()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


class TitanMemorySystem:
    """
    Main interface for the Titan Memory system.
    
    This provides high-level methods for using the memory model with
    the LLaDA GUI.
    """

    def __init__(self, config: TitanMemoryConfig = None):
        """Initialize the memory system.
        
        Args:
            config: Optional configuration
        """
        self.config = config or TitanMemoryConfig()
        self.model = TitanMemoryModel(self.config)
        self.initialized = True

        # Create default data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Default model path
        self.default_model_path = os.path.join(self.data_dir, "titan_memory_model.json")

        # Try to load existing model
        if os.path.exists(self.default_model_path):
            try:
                self.model.load_model(self.default_model_path)
                print(f"Loaded existing memory model from {self.default_model_path}")
            except Exception as e:
                print(f"Failed to load existing model: {str(e)}")
                print("Initializing new memory model instead")

    def forward_pass(self, input_vector: Union[np.ndarray, List[float], torch.Tensor]) -> Dict[str, Any]:
        """Run a forward pass through the memory model.
        
        Args:
            input_vector: Input vector
            
        Returns:
            Dictionary with predicted, new_memory, and surprise values
        """
        # Convert input to tensor
        if isinstance(input_vector, np.ndarray):
            x = torch.from_numpy(input_vector).float()
        elif isinstance(input_vector, list):
            x = torch.tensor(input_vector, dtype=torch.float32)
        elif isinstance(input_vector, torch.Tensor):
            x = input_vector.float()
        else:
            raise TypeError(f"Unsupported input type: {type(input_vector)}")

        # Run forward pass
        with torch.no_grad():
            result = self.model.forward(x)

        # Convert tensors to lists for easier handling
        return {
            "predicted": result["predicted"].detach().cpu().numpy().tolist(),
            "newMemory": result["new_memory"].detach().cpu().numpy().tolist(),
            "surprise": float(result["surprise"].item())
        }

    def train_step(self,
                   current_vector: Union[np.ndarray, List[float]],
                   next_vector: Union[np.ndarray, List[float]]) -> float:
        """Train the memory on a transition.
        
        Args:
            current_vector: Current state vector
            next_vector: Next state vector
            
        Returns:
            Loss value
        """
        # Convert inputs to tensors
        if isinstance(current_vector, np.ndarray):
            x_t = torch.from_numpy(current_vector).float()
        elif isinstance(current_vector, list):
            x_t = torch.tensor(current_vector, dtype=torch.float32)
        else:
            x_t = current_vector.float()

        if isinstance(next_vector, np.ndarray):
            x_next = torch.from_numpy(next_vector).float()
        elif isinstance(next_vector, list):
            x_next = torch.tensor(next_vector, dtype=torch.float32)
        else:
            x_next = next_vector.float()

        # Perform training step
        loss = self.model.train_step(x_t, x_next)

        # Save model after training
        try:
            # Properly detach tensors before saving
            self.model.save_model(self.default_model_path)
        except Exception as e:
            import traceback
            print(f"Warning: Failed to save model after training: {str(e)}")
            traceback.print_exc()

        return loss

    def train_sequence(self, sequence: List[Union[np.ndarray, List[float]]]) -> List[float]:
        """Train the memory on a sequence of vectors.
        
        Args:
            sequence: List of vectors
            
        Returns:
            List of loss values
        """
        if len(sequence) < 2:
            return []

        losses = []
        for i in range(len(sequence) - 1):
            loss = self.train_step(sequence[i], sequence[i + 1])
            losses.append(loss)

        return losses

    def update_memory(self, input_vector: Union[np.ndarray, List[float]]) -> Tuple[List[float], float]:
        """Update memory state based on input.
        
        Args:
            input_vector: Input vector
            
        Returns:
            Tuple of new memory state and surprise value
        """
        # Convert input to tensor
        if isinstance(input_vector, np.ndarray):
            x = torch.from_numpy(input_vector).float()
        elif isinstance(input_vector, list):
            x = torch.tensor(input_vector, dtype=torch.float32)
        else:
            x = input_vector.float()

        # Update memory
        new_memory, surprise = self.model.update_memory(x)

        # Make sure we properly detach tensors
        if isinstance(new_memory, torch.Tensor):
            return new_memory.detach().cpu().numpy().tolist(), surprise
        elif isinstance(new_memory, np.ndarray):
            return new_memory.tolist(), surprise
        else:
            return new_memory, surprise

    def get_memory_state(self) -> List[float]:
        """Get the current memory state.
        
        Returns:
            Memory state as a list of floats
        """
        memory = self.model.get_memory_state()
        if isinstance(memory, np.ndarray):
            return memory.tolist()
        elif isinstance(memory, torch.Tensor):
            return memory.detach().cpu().numpy().tolist()
        else:
            return memory

    def set_memory_state(self, state: Union[np.ndarray, List[float]]):
        """Set the memory state manually.
        
        Args:
            state: New memory state
        """
        self.model.set_memory_state(state)

    def reset_memory(self):
        """Reset the memory state to zeros."""
        self.model.reset_memory()

    def save_model(self, path: Optional[str] = None):
        """Save the model to a file.
        
        Args:
            path: Path to save file. If None, use default path.
        """
        save_path = path or self.default_model_path
        self.model.save_model(save_path)

    def load_model(self, path: Optional[str] = None):
        """Load the model from a file.
        
        Args:
            path: Path to model file. If None, use default path.
        """
        load_path = path or self.default_model_path
        if os.path.exists(load_path):
            self.model.load_model(load_path)
        else:
            raise FileNotFoundError(f"Model file not found: {load_path}")


# Simple testing function
def test_titan_memory():
    """Test the Titan Memory system with some simple inputs."""
    print("Testing Titan Memory System...")

    # Create system
    system = TitanMemorySystem()

    # Create some test vectors
    vec1 = np.random.randn(64)
    vec2 = np.random.randn(64)

    # Forward pass
    print("Running forward pass...")
    result = system.forward_pass(vec1)
    print(f"Surprise: {result['surprise']:.6f}")

    # Train step
    print("Running training step...")
    loss = system.train_step(vec1, vec2)
    print(f"Loss: {loss:.6f}")

    # Update memory
    print("Updating memory...")
    new_memory, surprise = system.update_memory(vec2)
    print(f"New surprise: {surprise:.6f}")

    print("Test complete!")
    return True


if __name__ == "__main__":
    test_titan_memory()
