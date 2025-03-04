### For 8GB VRAM GPUs:
- Enable 4-bit quantization
- Enable Extreme Memory Mode
- Generation Length: 32-48
- Sampling Steps: 32-128
- Block Length: 16

### For 12GB VRAM GPUs:
- Enable 8-bit or 4-bit quantization
- Enable Extreme Memory Mode
- Generation Length: 48-64
- Sampling Steps: 64-128
- Block Length: 32

### For 16GB+ VRAM GPUs:
- Use Normal or 8-bit precision
- Generation Length: 96-128
- Sampling Steps: 96-256
- Block Length: 32

### For CPU Mode:
- Generation Length: 32
- Sampling Steps: 16-32
- Block Length: 16

## Troubleshooting

If you encounter out-of-memory errors:
1. Reduce Generation Length first
2. Try 4-bit quantization if not already using it
3. Enable Extreme Memory Mode if not already enabled
4. Reduce steps if still having issues
5. If all else fails, switch to CPU mode