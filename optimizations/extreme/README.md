# LLaDA Extreme Memory Optimization

This module provides aggressive memory optimizations to allow running LLaDA on GPUs with limited VRAM (8-12GB). It implements several advanced techniques to dramatically reduce memory usage while maintaining functionality.

## Quick Start

To apply all extreme memory optimizations:

```bash
# Run the optimizer GUI
python optimize_extreme.py

# Or run directly from terminal
python optimizations/extreme/extreme_optimizer.py --apply-all
```

You can also use the desktop shortcut: `LLaDA_Extreme_Optimizer.desktop`

## How It Works

These optimizations use several aggressive techniques to reduce memory usage:

### 1. Model Pruning

- Removes less important attention heads (up to 30%)
- Prunes feed-forward network layers (up to 20%)
- Prioritizes keeping the most important model components

### 2. Progressive Loading

- Loads the model in small blocks instead of all at once
- Significantly reduces peak memory usage during initialization
- Automatically offloads less used layers to CPU

### 3. Optimized Diffusion Process

- Implements a memory-efficient diffusion algorithm
- Aggressively cleans up intermediate tensors
- Uses dynamic offloading of layers during generation

### 4. Memory Leak Patches

- Applies patches to fix memory leaks in PyTorch and transformers
- Forces more aggressive garbage collection
- Overrides memory-intensive operations with optimized versions

## Optimization Levels

The extreme optimizer offers three levels of optimization:

1. **Mild**: 
   - Applies basic memory optimizations
   - Reduces VRAM usage by ~20-30%
   - Minimal impact on generation quality

2. **Moderate** (Default): 
   - Applies more aggressive memory optimizations
   - Reduces VRAM usage by ~40-50%
   - Slight impact on generation quality

3. **Aggressive**: 
   - Applies the most extreme memory optimizations
   - Reduces VRAM usage by ~60-70%
   - May impact generation quality but enables use with 8GB GPUs

## Requirements

- NVIDIA GPU with CUDA support (optimized for 8-12GB VRAM)
- PyTorch 2.0 or later
- transformers 4.38.2
- bitsandbytes (for 4-bit quantization)

## Restoring Original Files

If you need to restore the original files:

```bash
python optimizations/extreme/extreme_optimizer.py --restore
```

Or use the "Restore Original Files" button in the GUI.

## Technical Details

For advanced users and developers:

- Model pruning is based on parameter importance determined by L2 norm
- The offloading manager dynamically moves layers between GPU and CPU memory 
- Progressive loading preserves the model architecture while reducing peak memory
- Memory patches override low-level PyTorch functions to reduce allocations

## Warning

These optimizations are very aggressive and may impact generation quality. They are designed to make LLaDA usable on lower-end hardware, not to maintain perfect quality. For best results, use a GPU with 16GB+ VRAM.
