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


50% load, 9%
4.02
parallel_model.fit(x_train, y_train, epochs=40, batch_size=512)#7168)
Epoch 39/40
98/98 [==============================] - 5s 54ms/step - loss: 0.2330 - accuracy: 0.9400
Epoch 40/40
98/98 [==============================] - 5s 53ms/step - loss: 0.1610 - accuracy: 0.9656

60% load,15% ram
2.35
parallel_model.fit(x_train, y_train, epochs=40, batch_size=1024)#7168)
Epoch 39/40
49/49 [==============================] - 3s 57ms/step - loss: 0.2216 - accuracy: 0.9517
Epoch 40/40
49/49 [==============================] - 3s 58ms/step - loss: 0.1297 - accuracy: 0.9752

70% load, 30% ram
1.55 time
parallel_model.fit(x_train, y_train, epochs=40, batch_size=2048)#7168)#7168)
Epoch 39/40
25/25 [==============================] - 2s 68ms/step - loss: 0.1953 - accuracy: 0.9420
Epoch 40/40
25/25 [==============================] - 2s 67ms/step - loss: 0.1396 - accuracy: 0.9647

1.40
75% load, 40% ram
parallel_model.fit(x_train, y_train, epochs=40, batch_size=4096)#7168)#7168)
Epoch 39/40
13/13 [==============================] - 1s 104ms/step - loss: 0.1160 - accuracy: 0.9671
Epoch 40/40
13/13 [==============================] - 1s 104ms/step - loss: 0.1239 - accuracy: 0.9651

1.45
80% load, 50% ram
parallel_model.fit(x_train, y_train, epochs=40, batch_size=5120)#7168)#7168)
Epoch 39/40
7/7 [==============================] - 1s 175ms/step - loss: 0.2035 - accuracy: 0.9356
Epoch 40/40
7/7 [==============================] - 1s 171ms/step - loss: 0.1847 - accuracy: 0.9429


85% load, 50% ram
parallel_model.fit(x_train, y_train, epochs=40, batch_size=7168)#7168)
Epoch 40/40
7/7 [==============================] - 1s 173ms/step - loss: 0.1083 - accuracy: 0.9699


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

## Google Cloud - Dual L4 GPU VM
### TensforFlow / Keras test ML training run
Run a standard concurrent saturation TensorFlow/Keras ML job from U of Toronto to check batch size optimums under 30 epochs to get close to 1.0 fitness - 25 avoids overfit
```


base) michael@l4-4-2:~$ git clone https://github.com/ObrienlabsDev/machine-learning.git
(base) michael@l4-4-2:~/machine-learning$ vi environments/windows/src/tflow.py 
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

with strategy.scope():
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
# https://keras.io/api/models/model/
  parallel_model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# https://keras.io/api/models/model_training_apis/
  parallel_model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
parallel_model.fit(x_train, y_train, epochs=30, batch_size=2048)#5120)#7168)#7168)

(base) michael@l4-4-2:~/machine-learning$ cat environments/windows/Dockerfile 
FROM tensorflow/tensorflow:latest-gpu
WORKDIR /src
COPY /src/tflow.py .
CMD ["python", "tflow.py"]

base) michael@l4-4-2:~/machine-learning$ ./build.sh 
Sending build context to Docker daemon  6.656kB
Step 1/4 : FROM tensorflow/tensorflow:latest-gpu
latest-gpu: Pulling from tensorflow/tensorflow

successfully tagged ml-tensorflow-win:latest
2023-11-30 20:29:26.443809: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-11-30 20:29:26.497571: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-11-30 20:29:26.497614: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-11-30 20:29:26.499104: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2023-11-30 20:29:26.506731: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-11-30 20:29:31.435829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 20795 MB memory:  -> device: 0, name: NVIDIA L4, pci bus id: 0000:00:03.0, compute capability: 8.9
2023-11-30 20:29:31.437782: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 20795 MB memory:  -> device: 1, name: NVIDIA L4, pci bus id: 0000:00:04.0, compute capability: 8.9
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
169001437/169001437 [==============================] - 3s 0us/step
Epoch 1/30

023-11-30 20:30:19.985861: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
2023-11-30 20:30:20.001134: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
2023-11-30 20:30:29.957119: I external/local_xla/xla/service/service.cc:168] XLA service 0x7f9c6bf3a4f0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2023-11-30 20:30:29.957184: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA L4, Compute Capability 8.9
2023-11-30 20:30:29.957192: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (1): NVIDIA L4, Compute Capability 8.9
2023-11-30 20:30:29.965061: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1701376230.063893      80 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

25/25 [==============================] - 71s 317ms/step - loss: 4.9465 - accuracy: 0.0418
Epoch 2/30
25/25 [==============================] - 4s 142ms/step - loss: 3.8430 - accuracy: 0.1214
Epoch 3/30
25/25 [==============================] - 4s 142ms/step - loss: 3.3694 - accuracy: 0.1967
Epoch 4/30
25/25 [==============================] - 4s 143ms/step - loss: 3.0832 - accuracy: 0.2544
Epoch 5/30
25/25 [==============================] - 4s 143ms/step - loss: 2.7049 - accuracy: 0.3326
Epoch 6/30
25/25 [==============================] - 4s 143ms/step - loss: 2.3329 - accuracy: 0.4119
Epoch 7/30
25/25 [==============================] - 4s 143ms/step - loss: 1.9781 - accuracy: 0.4824
Epoch 8/30
25/25 [==============================] - 4s 143ms/step - loss: 1.9177 - accuracy: 0.4948
Epoch 9/30
25/25 [==============================] - 4s 142ms/step - loss: 1.4980 - accuracy: 0.5937
Epoch 10/30
25/25 [==============================] - 4s 144ms/step - loss: 1.3247 - accuracy: 0.6322
Epoch 11/30
25/25 [==============================] - 4s 142ms/step - loss: 1.0408 - accuracy: 0.7063
Epoch 12/30
25/25 [==============================] - 4s 142ms/step - loss: 0.9150 - accuracy: 0.7439
Epoch 13/30
25/25 [==============================] - 4s 143ms/step - loss: 0.8210 - accuracy: 0.7648
Epoch 14/30
25/25 [==============================] - 4s 142ms/step - loss: 0.5581 - accuracy: 0.8424
Epoch 15/30
25/25 [==============================] - 4s 141ms/step - loss: 0.4635 - accuracy: 0.8709
Epoch 16/30
25/25 [==============================] - 4s 142ms/step - loss: 0.4771 - accuracy: 0.8610
Epoch 17/30
25/25 [==============================] - 4s 142ms/step - loss: 0.9404 - accuracy: 0.7228
Epoch 18/30
25/25 [==============================] - 4s 143ms/step - loss: 0.5478 - accuracy: 0.8385
Epoch 19/30
25/25 [==============================] - 4s 143ms/step - loss: 0.4107 - accuracy: 0.8867
Epoch 20/30
25/25 [==============================] - 4s 143ms/step - loss: 0.2424 - accuracy: 0.9345
Epoch 21/30
25/25 [==============================] - 4s 146ms/step - loss: 0.1677 - accuracy: 0.9587
Epoch 22/30
25/25 [==============================] - 4s 142ms/step - loss: 0.1419 - accuracy: 0.9659
Epoch 23/30
25/25 [==============================] - 4s 141ms/step - loss: 0.1861 - accuracy: 0.9510
Epoch 24/30
25/25 [==============================] - 4s 141ms/step - loss: 0.2771 - accuracy: 0.9264
Epoch 25/30
25/25 [==============================] - 4s 142ms/step - loss: 0.2663 - accuracy: 0.9326
Epoch 26/30
25/25 [==============================] - 4s 141ms/step - loss: 0.1710 - accuracy: 0.9600
Epoch 27/30
25/25 [==============================] - 4s 141ms/step - loss: 0.4977 - accuracy: 0.8626
Epoch 28/30
25/25 [==============================] - 4s 141ms/step - loss: 0.6559 - accuracy: 0.8100
Epoch 29/30
25/25 [==============================] - 4s 143ms/step - loss: 0.3074 - accuracy: 0.9105
Epoch 30/30
25/25 [==============================] - 4s 143ms/step - loss: 0.1834 - accuracy: 0.9515
(base) michael@l4-4-2:~/machine-learning$ 
```
<img width="898" alt="Screenshot 2023-11-30 at 15 31 16" src="https://github.com/GoogleCloudPlatform/pubsec-declarative-toolkit/assets/24765473/ba065148-45ec-40de-a532-72644193c41a">


## GPU: Dual RTX-A4500 Ampere without NVLink in Z790H 64G i9-14900K
### Power:
### Temp:
- liquid temp
- CPU temp 
- RAM temp outside
- Board power cap outside
- 
### Code
```
```
### Results
```
```

## GPU: Dual RTX-A4500 Ampere with NVLink in Z790H 64G i9-14900K
## CPU: Z790H 64G i9-14900K
### Power:
### Temp:
- PSU power in-internal: 350W-470W peak
- liquid temp = 38
- CPU temp = 71-76
- RAM temp outside
- RAM temp software: 61C
- Board power cap outside
- CPU wsl2 96-99%
![image](https://github.com/ObrienlabsDev/machine-learning/assets/24765473/de59179c-95b0-40cb-afa3-c7a05eb86d3a)

### Code
```
strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
parallel_model.fit(x_train, y_train, epochs=25, batch_size=256)

```
### Results
```
196/196 [==============================] - 61s 285ms/step - loss: 4.3420 - accuracy: 0.0941
Epoch 2/25
196/196 [==============================] - 55s 281ms/step - loss: 3.6117 - accuracy: 0.1755
Epoch 3/25
2023-12-08 21:15:54.565643: E external/local_xla/xla/stream_executor/stream_executor_internal.h:177] SetPriority unimplemented for this stream.
196/196 [==============================] - 55s 281ms/step - loss: 3.3433 - accuracy: 0.2257
Epoch 4/25
196/196 [==============================] - 55s 281ms/step - loss: 3.3183 - accuracy: 0.2407
Epoch 5/25
196/196 [==============================] - 56s 283ms/step - loss: 2.8984 - accuracy: 0.2995
Epoch 6/25
196/196 [==============================] - 57s 291ms/step - loss: 2.7675 - accuracy: 0.3311
Epoch 7/25
196/196 [==============================] - 56s 286ms/step - loss: 3.0207 - accuracy: 0.3210
Epoch 8/25
196/196 [==============================] - 56s 288ms/step - loss: 3.9374 - accuracy: 0.1562
Epoch 9/25
196/196 [==============================] - 57s 289ms/step - loss: 3.5116 - accuracy: 0.1987
Epoch 10/25
196/196 [==============================] - 57s 290ms/step - loss: 3.0358 - accuracy: 0.2711
Epoch 11/25
196/196 [==============================] - 57s 292ms/step - loss: 2.8239 - accuracy: 0.3067
Epoch 12/25
196/196 [==============================] - 57s 291ms/step - loss: 2.6456 - accuracy: 0.3391
Epoch 13/25
196/196 [==============================] - 55s 281ms/step - loss: 2.5414 - accuracy: 0.3609
Epoch 14/25
196/196 [==============================] - 54s 278ms/step - loss: 2.3249 - accuracy: 0.4089
Epoch 15/25
196/196 [==============================] - 55s 279ms/step - loss: 2.1558 - accuracy: 0.4413
Epoch 16/25
196/196 [==============================] - 54s 277ms/step - loss: 2.0078 - accuracy: 0.4763
Epoch 17/25
196/196 [==============================] - 54s 276ms/step - loss: 1.8048 - accuracy: 0.5175
Epoch 18/25
196/196 [==============================] - 56s 288ms/step - loss: 1.6319 - accuracy: 0.5605
Epoch 19/25
196/196 [==============================] - 55s 283ms/step - loss: 1.4296 - accuracy: 0.6122
Epoch 20/25
196/196 [==============================] - 55s 278ms/step - loss: 1.2125 - accuracy: 0.6641
Epoch 21/25
196/196 [==============================] - 54s 277ms/step - loss: 1.0231 - accuracy: 0.7140
Epoch 22/25
196/196 [==============================] - 54s 278ms/step - loss: 0.9759 - accuracy: 0.7312
Epoch 23/25
196/196 [==============================] - 54s 278ms/step - loss: 1.1977 - accuracy: 0.6877
Epoch 24/25
196/196 [==============================] - 54s 278ms/step - loss: 1.1991 - accuracy: 0.6801
Epoch 25/25
196/196 [==============================] - 55s 279ms/step - loss: 0.6931 - accuracy: 0.8060

```

## GPU: Dual RTX-4090 Suprim Liquid X Ada without NVLink in Z790H 192G i9-13900K
### Power:
### Temp:
- liquid temp
- CPU temp 
- RAM temp outside
- Board power cap outside
- 
### Code
```
```
### Results
```
```

## CPU: Z790H 192G i9-13900K
### Power:
### Temp:
- liquid temp
- CPU temp 
- RAM temp outside
- Board power cap outside
- 
### Code
```
```
### Results
```
```



