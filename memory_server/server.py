#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple memory server implementation for the LLaDA GUI.

This provides a lightweight HTTP API for the Titan memory system, fully
integrated within the LLaDA GUI project.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from flask import Flask, request, jsonify

# Configuration class for memory model
@dataclass
class TitanMemoryConfig:
    """Configuration for the Titan Memory model."""
    input_dim: int = 64         # Input dimension
    output_dim: int = 64        # Memory state dimension
    hidden_dim: int = 32        # Hidden layer dimension
    learning_rate: float = 1e-3 # Learning rate for optimizer
    forget_gate_init: float = 0.01 # Initial forget gate value

# Memory model implementation
class TitanMemoryModel(nn.Module):
    """
    Neural memory model based on the Titan Memory system.
    
    This model maintains a memory state and can predict the next token based
    on the current input and memory state.
    """
    
    def __init__(self, config: TitanMemoryConfig = None):
        """Initialize the model.
        
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
        self.output_dim = config.output_dim
        
        # Initialize layers
        self.fc1 = nn.Linear(self.input_dim + self.output_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_dim + self.output_dim)
        self.forget_gate = nn.Parameter(torch.tensor(config.forget_gate_init))
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # Initialize memory state
        self.reset_memory()
    
    def reset_memory(self):
        """Reset the memory state to zeros."""
        self.memory_state = torch.zeros(self.output_dim)
    
    def get_memory_state(self) -> np.ndarray:
        """Get the current memory state."""
        return self.memory_state.detach().cpu().numpy()
    
    def set_memory_state(self, state: Union[np.ndarray, torch.Tensor, List[float]]):
        """Set the memory state manually."""
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
        forget_value = torch.sigmoid(self.forget_gate)
        gated_memory = memory * (1 - forget_value)
        
        # Combine input and gated memory
        combined = torch.cat([x, gated_memory], dim=0)
        
        # MLP forward pass
        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)
        
        # Split output into new memory and predicted next input
        new_memory = output[:self.output_dim]
        predicted = output[self.output_dim:]
        
        # Calculate surprise (MSE between predicted and actual input)
        diff = predicted - x
        surprise = torch.mean(diff * diff)
        
        return {
            "predicted": predicted,
            "new_memory": new_memory,
            "surprise": surprise
        }
    
    def train_step(self, x_t: torch.Tensor, x_next: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Perform a training step.
        
        Args:
            x_t: Current input tensor
            x_next: Next input tensor (target)
            memory: Current memory state
            
        Returns:
            Loss value tensor
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        result = self.forward(x_t, memory)
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
        
        return total_loss
    
    def save_model(self, path: str):
        """Save model to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state and config
        state_dict = self.state_dict()
        config_dict = vars(self.config)
        save_data = {
            "state_dict": {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                          for k, v in state_dict.items()},
            "config": config_dict,
            "memory_state": self.memory_state.cpu().numpy().tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_model(self, path: str):
        """Load model from file."""
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        # Load config
        config_dict = save_data.get("config", {})
        self.config = TitanMemoryConfig(**config_dict)
        
        # Initialize tensor parameters
        state_dict = {}
        for k, v in save_data["state_dict"].items():
            if isinstance(v, list):
                state_dict[k] = torch.tensor(v)
            else:
                state_dict[k] = v
        
        self.load_state_dict(state_dict)
        
        # Load memory state
        memory_state = save_data.get("memory_state", None)
        if memory_state:
            self.memory_state = torch.tensor(memory_state)
        else:
            self.reset_memory()

# Flask server for HTTP API
app = Flask(__name__)
model = None
memory_state = None

@app.route('/status', methods=['GET'])
def get_status():
    """Get model status."""
    if model is None:
        return jsonify({"status": "No model initialized"})
    return jsonify(vars(model.config))

@app.route('/init', methods=['POST'])
def init_model():
    """Initialize model."""
    global model, memory_state
    
    try:
        data = request.json or {}
        input_dim = data.get('inputDim', 64)
        output_dim = data.get('outputDim', 64)
        
        config = TitanMemoryConfig(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Initialize model
        model = TitanMemoryModel(config)
        
        return jsonify({
            "message": "Model initialized", 
            "config": vars(model.config)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forward', methods=['POST'])
def forward_pass():
    """Run forward pass through model."""
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    try:
        data = request.json
        if not data or 'x' not in data:
            return jsonify({"error": "Missing input vector"}), 400
        
        x = torch.tensor(data['x'], dtype=torch.float32)
        
        # Run forward pass
        with torch.no_grad():
            result = model.forward(x)
        
        # Extract values
        predicted = result["predicted"].detach().cpu().numpy().tolist()
        new_memory = result["new_memory"].detach().cpu().numpy().tolist()
        surprise = float(result["surprise"].item())
        
        # Update memory state
        model.memory_state = result["new_memory"].detach().clone()
        
        return jsonify({
            "predicted": predicted,
            "memory": new_memory,
            "surprise": surprise
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/trainStep', methods=['POST'])
def train_step():
    """Perform training step."""
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    try:
        data = request.json
        if not data or 'x_t' not in data or 'x_next' not in data:
            return jsonify({"error": "Missing required parameters"}), 400
        
        x_t = torch.tensor(data['x_t'], dtype=torch.float32)
        x_next = torch.tensor(data['x_next'], dtype=torch.float32)
        
        # Run training step
        cost = model.train_step(x_t, x_next, model.memory_state)
        
        # Run forward pass for results
        with torch.no_grad():
            result = model.forward(x_t)
        
        # Extract values
        predicted = result["predicted"].detach().cpu().numpy().tolist()
        cost_val = float(cost.item())
        surprise = float(result["surprise"].item())
        
        return jsonify({
            "cost": cost_val,
            "predicted": predicted,
            "surprise": surprise
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save', methods=['POST'])
def save_model():
    """Save model to file."""
    global model
    
    if model is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    try:
        data = request.json
        if not data or 'path' not in data:
            return jsonify({"error": "Missing path parameter"}), 400
        
        model.save_model(data['path'])
        return jsonify({"message": "Model saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load', methods=['POST'])
def load_model():
    """Load model from file."""
    global model
    
    if model is None:
        model = TitanMemoryModel()
    
    try:
        data = request.json
        if not data or 'path' not in data:
            return jsonify({"error": "Missing path parameter"}), 400
        
        model.load_model(data['path'])
        return jsonify({"message": "Model loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_server(host='127.0.0.1', port=3000):
    """Start the server."""
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_server()
