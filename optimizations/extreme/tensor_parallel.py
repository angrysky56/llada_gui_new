#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tensor parallelism for LLaDA GUI.
"""

import torch
import torch.nn as nn
import logging
import warnings

logger = logging.getLogger(__name__)

class TensorParallelWrapper:
    """
    Wrapper to enable tensor parallelism for LLaDA model.
    This splits model computation across multiple GPUs.
    """
    
    def __init__(self, model, num_gpus=None):
        """
        Initialize tensor parallelism.
        
        Args:
            model: The model to parallelize
            num_gpus: Number of GPUs to use (default: use all available)
        """
        self.model = model
        self.original_model = model
        self.num_gpus = num_gpus or torch.cuda.device_count()
        
        if self.num_gpus <= 1:
            logger.warning("Only one GPU available, disabling tensor parallelism")
            return
        
        # Initialize process group if not already done
        try:
            self._init_process_group()
        except Exception as e:
            logger.error(f"Failed to initialize process group: {e}")
            logger.warning("Disabling tensor parallelism")
            self.num_gpus = 1
            return
            
        # Split the model across GPUs
        self.parallelize_model()
    
    def _init_process_group(self):
        """Initialize distributed process group for NCCL."""
        import torch.distributed as dist
        if not dist.is_initialized():
            # Initialize process group with NCCL backend
            dist.init_process_group(backend='nccl')
    
    def parallelize_model(self):
        """Split model across multiple GPUs."""
        try:
            # Find linear and attention layers to parallelize
            layers_to_parallelize = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.size(0) > 1024:
                    layers_to_parallelize.append((name, module))
            
            # Sort layers by size (largest first)
            layers_to_parallelize.sort(key=lambda x: x[1].weight.size(0) * x[1].weight.size(1), reverse=True)
            
            # Assign layers to GPUs
            gpu_assignments = {}
            layer_counts = [0] * self.num_gpus
            
            for i, (name, module) in enumerate(layers_to_parallelize):
                # Assign to GPU with fewest layers
                target_gpu = layer_counts.index(min(layer_counts))
                gpu_assignments[name] = target_gpu
                layer_counts[target_gpu] += 1
                
                # Move layer to assigned GPU
                module.to(f'cuda:{target_gpu}')
            
            # Store GPU assignments
            self.gpu_assignments = gpu_assignments
            
            # Change the model's forward method to use our custom implementation
            self.original_forward = self.model.forward
            self.model.forward = self.forward
            
            logger.info(f"Model parallelized across {self.num_gpus} GPUs")
            logger.info(f"Layer distribution: {layer_counts}")
            
        except Exception as e:
            logger.error(f"Error during model parallelization: {e}")
            logger.warning("Disabling tensor parallelism")
            # Restore original model
            self.model.forward = self.original_model.forward
            self.model = self.original_model
    
    def forward(self, input_ids, **kwargs):
        """
        Custom forward pass with tensor parallelism.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments for the model
            
        Returns:
            Model outputs
        """
        # If not parallelized, fall back to original forward
        if self.num_gpus <= 1 or not hasattr(self, 'gpu_assignments'):
            return self.original_forward(input_ids, **kwargs)
        
        try:
            # Ensure input is on first GPU
            input_ids = input_ids.to('cuda:0')
            
            # Prepare additional kwargs
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to('cuda:0')
            
            # Get embeddings from first GPU
            if hasattr(self.model, 'get_input_embeddings'):
                embeddings = self.model.get_input_embeddings()(input_ids)
            else:
                # Fall back to original forward
                logger.warning("Model doesn't have get_input_embeddings, using original forward")
                return self.original_forward(input_ids, **kwargs)
            
            # Find model layers
            layers = None
            for attribute in ["layers", "encoder.layer", "transformer.h", "model.layers"]:
                try:
                    current = self.model
                    for part in attribute.split('.'):
                        current = getattr(current, part)
                    if isinstance(current, (list, torch.nn.ModuleList)):
                        layers = current
                        break
                except (AttributeError, TypeError):
                    continue
            
            if layers is None:
                logger.warning("Could not find model layers, using original forward")
                return self.original_forward(input_ids, **kwargs)
            
            # Process each layer with tensor parallelism
            hidden_states = embeddings
            
            for layer_idx, layer in enumerate(layers):
                # Find assigned GPU for this layer
                layer_name = f'layers.{layer_idx}'
                target_gpu = self.gpu_assignments.get(layer_name, 0)
                
                # Move hidden states to target GPU
                hidden_states = hidden_states.to(f'cuda:{target_gpu}')
                
                # Process layer
                hidden_states = layer(hidden_states)
                
                # Move back to first GPU for next layer if needed
                if layer_idx < len(layers) - 1:
                    next_layer_name = f'layers.{layer_idx + 1}'
                    next_gpu = self.gpu_assignments.get(next_layer_name, 0)
                    
                    if next_gpu != target_gpu:
                        hidden_states = hidden_states.to(f'cuda:{next_gpu}')
            
            # Move to first GPU for final processing
            hidden_states = hidden_states.to('cuda:0')
            
            # Final output processing
            if hasattr(self.model, 'head'):
                outputs = self.model.head(hidden_states)
            elif hasattr(self.model, 'lm_head'):
                outputs = self.model.lm_head(hidden_states)
            else:
                # Fall back to original forward for unknown structure
                logger.warning("Unknown model structure, falling back to original forward")
                return self.original_forward(input_ids, **kwargs)
            
            # Create output structure similar to original model
            # This part depends on the model structure, may need adjustment
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.logits = outputs
            
            return result
            
        except Exception as e:
            logger.error(f"Error during tensor parallel forward: {e}")
            logger.warning("Falling back to original forward")
            return self.original_forward(input_ids, **kwargs)

def is_tensor_parallelism_available():
    """
    Check if tensor parallelism is available on this system.
    
    Returns:
        bool: True if available, False otherwise
    """
    # Check if multiple GPUs are available
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    
    # Check if NCCL is available
    try:
        import torch.distributed as dist
        return dist.is_nccl_available()
    except:
        return False
