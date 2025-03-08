#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for the memory server.
"""

# Server configuration
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 3000

# Model configuration
DEFAULT_MODEL_CONFIG = {
    'input_dim': 64,
    'output_dim': 64,
    'hidden_dim': 32,
    'learning_rate': 1e-3,
    'forget_gate_init': 0.01
}

# Auto-initialization of the model when server starts
AUTO_INIT_MODEL = True

# Default model path
import os

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/default_memory_model.json')
