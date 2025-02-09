import tensorflow as tf
<<<<<<< HEAD
#tf.compat.v1.disable_eager_execution()
#print(tf.__version__)
=======
# 36 - optimize for OSX cpu (not gpu) parallelism
import os
# M4 Max
os.environ["OMP_NUM_THREADS"] = "16"
# M1 Max
os.environ["OMP_NUM_THREADS"] = "10"
tf.config.threading.set_inter_op_parallelism_threads(64)
tf.config.threading.set_intra_op_parallelism_threads(64)

>>>>>>> refs/remotes/origin/main
#import keras
#from keras.utils import multi_gpu_model
#import keras.backend as k
#https://github.com/microsoft/tensorflow-directml/issues/352

# https://www.tensorflow.org/guide/distributed_training
#
# https://www.tensorflow.org/tutorials/distribute/keras
# https://keras.io/guides/distributed_training/
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#NUM_GPUS = 2
#strategy = tf.contrib.distribute.MirroredStrategy()#num_gpus=NUM_GPUS)
# working on dual RTX-4090
#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#WARNING:tensorflow:Some requested devices in `tf.distribute.Strategy` are not visible to TensorFlow: /replica:0/task:0/device:GPU:1,/replica:0/task:0/device:GPU:0
#Number of devices: 2

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

#central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
#strategy = tf.distribute.MultiWorkerMirroredStrategy() # not in tf 1.5
#print("mirrored_strategy: ",mirrored_strategy)
#strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"],cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(all_reduce_alg="hierarchical_copy"))
#mirrored_strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# https://learn.microsoft.com/en-us/windows/ai/directml/gpu-faq
#a = tf.constant([1.])
#b = tf.constant([2.])
#c = tf.add(a, b)

#gpu_config = tf.GPUOptions()
#gpu_config.visible_device_list = "1"#"0,1"
#gpu_config.visible_device_list = "0,1"
#gpu_config.allow_growth=True

#session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))
#print(session.run(c))
#tensorflow.python.framework.errors_impl.AlreadyExistsError: TensorFlow device (DML:0) is being mapped to multiple DML devices (0 now, and 1 previously), which is not supported. This may be the result of providing different GPU configurations (ConfigProto.gpu_options, for example different visible_device_list) when creating multiple Sessions in the same process. This is not  currently supported, see https://github.com/tensorflow/tensorflow/issues/19083
#from keras import backend as K
#K.set_session(session)

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

with strategy.scope():
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
# https://keras.io/api/models/model/
  parallel_model = tf.keras.applications.ResNet50(
#model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)
# https://saturncloud.io/blog/how-to-do-multigpu-training-with-keras/  
  #parallel_model = multi_gpu_model(model, gpus=2)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# https://keras.io/api/models/model_training_apis/
  parallel_model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
parallel_model.fit(x_train, y_train, epochs=25, batch_size=2048)#5120)#7168)#7168)
