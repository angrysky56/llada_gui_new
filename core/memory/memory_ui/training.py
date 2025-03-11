#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training components for memory system.

This module provides training functionality for the memory system.
"""

import numpy as np
import requests
from PyQt6.QtCore import QThread, pyqtSignal, QThread

class TrainingThread(QThread):
    """Thread for training the memory model."""

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, memory_interface, prompt, generated_text):
        """Initialize the thread.
        
        Args:
            memory_interface: Memory interface to use
            prompt: Input prompt
            generated_text: Generated text output
        """
        super().__init__()
        self.memory_interface = memory_interface
        self.prompt = prompt
        self.generated_text = generated_text

    def run(self):
        """Run the training process."""
        try:
            # Encode the text to simple embedding vectors
            # In a real implementation, this would use proper embeddings
            # This is a simplified version that just uses character counts

            # Function to create a simple embedding
            def simple_embed(text, dim=64):
                # Normalize and pad/truncate to the desired dimension
                char_counts = np.zeros(dim)
                for i, char in enumerate(text[:1000]):
                    char_counts[i % dim] += ord(char) % 10
                # Normalize
                norm = np.linalg.norm(char_counts)
                if norm > 0:
                    char_counts = char_counts / norm
                return char_counts

            # Create embeddings
            prompt_vec = simple_embed(self.prompt)
            gen_vec = simple_embed(self.generated_text)

            # Train in several steps for progress visualization
            steps = 5
            for i in range(steps):
                # Send training request to API
                response = requests.post(
                    f"{self.memory_interface.api_url}/trainStep",
                    json={
                        "x_t": prompt_vec.tolist(),
                        "x_next": gen_vec.tolist()
                    },
                    timeout=10
                )
                response.raise_for_status()

                # Update progress
                self.progress.emit(int((i + 1) / steps * 100))

                # Small delay to show progress
                QThread.msleep(500)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
