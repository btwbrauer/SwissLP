import torch
import transformers
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
if hasattr(torch.version, 'hip'):
    print(f"ROCm version: {torch.version.hip}")
else:
    print("ROCm version: N/A")


