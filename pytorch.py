# https://github.com/ObrienlabsDev/machine-learning/issues/49
import torch
import os

print("cpus: ", os.cpu_count())

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

