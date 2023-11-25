# machine-learning
Machine Learning - AI - Tensorflow - Keras - NVidia - Google

# Environments

## RTX-4090 Suprim Dual
```
michael@13900b MINGW64 /c/wse_github/obrienlabsdev/machine-learning/environments/windows (main)
$ docker build -t ml-tensorflow .
michael@13900b MINGW64 /c/wse_github/obrienlabsdev/machine-learning/environments/windows (main)
$ docker run --rm --gpus all ml-tensorflow

2023-11-25 04:06:47.732831: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21286 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:01:00.0, compute capability: 8.9
2023-11-25 04:06:47.733361: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:880] could not open file to read NUMA node: /sys/bus/pci/devices/0000:02:00.0/numa_node
Your kernel may have been built without NUMA support.
2023-11-25 04:06:47.733383: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 21286 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:02:00.0, compute capability: 8.9
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
169001437/169001437 [==============================] - 8s 0us/step
Epoch 1/40
2023-11-25 04:07:16.813872: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-11-25 04:07:17.564042: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-11-25 04:07:19.761064: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f15381ed280 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2023-11-25 04:07:19.761096: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce RTX 4090, Compute Capability 8.9
2023-11-25 04:07:19.761099: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (1): NVIDIA GeForce RTX 4090, Compute Capability 8.9
2023-11-25 04:07:19.764829: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2023-11-25 04:07:19.821529: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
196/196 [==============================] - 37s 53ms/step - loss: 4.3691 - accuracy: 0.0853
Epoch 2/40
196/196 [==============================] - 9s 48ms/step - loss: 3.8850 - accuracy: 0.1505

```
## RTX-A4500 - Dual

## Quadro RTX-5000 - TU104
```
2023-10-04 00:07:23.774558: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1886] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14045 MB memory:  -> device: 0, name: Quadro RTX 5000, pci bus id: 0000:01:00.0, compute capability: 7.5
Epoch 1/40
2023-10-04 00:07:36.487253: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8600
2023-10-04 00:07:38.967127: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x32e96c70 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2023-10-04 00:07:38.967167: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Quadro RTX 5000, Compute Capability 7.5

```

## RTX-3500 ADA in Lenovo P1 Gen 6 - 2023

