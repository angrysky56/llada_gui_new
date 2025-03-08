#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory offloading manager for LLaDA GUI.
"""

import torch


class OffloadingManager:
    """Manages CPU-GPU memory offloading for LLaDA model."""

    def __init__(self, model, device="cuda", offload_threshold=0.8):
        """
        Initialize the offloading manager.
        
        Args:
            model: The LLaDA model
            device: Device to use for computation
            offload_threshold: GPU memory threshold for offloading (0.0-1.0)
        """
        self.model = model
        self.device = device
        self.threshold = offload_threshold
        self.layer_devices = {}  # Track where each layer is stored

        # Store layer references for quick access
        self.layers = []
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in ['attention', 'mlp', 'transformer.h']):
                self.layers.append((name, module))
                self.layer_devices[name] = device

    def check_memory(self):
        """Check GPU memory usage and offload if needed."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return

        # Get current memory usage
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
        free = total - used
        usage = used / total

        # Offload if memory usage is above threshold
        if usage > self.threshold:
            self._offload_layers(int((usage - self.threshold) * len(self.layers)))

    def _offload_layers(self, num_layers):
        """Offload layers to CPU to free memory."""
        # Sort layers by importance (offload attention first, then others)
        offload_candidates = sorted(
            [(name, module) for name, module in self.layers if self.layer_devices[name] == "cuda"],
            key=lambda x: 'attention' not in x[0]
        )

        # Offload the specified number of layers
        for i, (name, module) in enumerate(offload_candidates):
            if i >= num_layers:
                break

            # Move layer to CPU
            module.to("cpu")
            self.layer_devices[name] = "cpu"

            # Clear GPU cache
            torch.cuda.empty_cache()

    def ensure_layer_on_device(self, layer_name, device=None):
        """Ensure a layer is on the specified device."""
        if device is None:
            device = self.device

        # Find the layer
        for name, module in self.layers:
            if name == layer_name:
                # Only move if not already on the target device
                if self.layer_devices[name] != device:
                    module.to(device)
                    self.layer_devices[name] = device
                return True

        return False

    def prepare_for_inference(self, layer_names=None):
        """Prepare model for inference by loading required layers to GPU."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return

        # If specific layers not provided, load all
        if layer_names is None:
            # Load most important layers first
            sorted_layers = sorted(
                [(name, module) for name, module in self.layers],
                key=lambda x: 0 if 'attention' in x[0] else 1
            )

            for name, module in sorted_layers:
                # Load until memory gets tight
                if self.layer_devices[name] != "cuda":
                    # Check if we have space
                    total = torch.cuda.get_device_properties(0).total_memory
                    used = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
                    usage = used / total

                    if usage > self.threshold:
                        break

                    # Move to GPU
                    module.to("cuda")
                    self.layer_devices[name] = "cuda"
        else:
            # Load specific layers
            for name in layer_names:
                self.ensure_layer_on_device(name, "cuda")
