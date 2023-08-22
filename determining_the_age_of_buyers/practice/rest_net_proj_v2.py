"""
   file to send to the server
"""
# imports
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Dense,
                                     GlobalAveragePooling2D,
                                     )

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd


def load_data(path: str, train: bool = False) -> iter:
    """
    generating samples from a dataset for training
    and checking the quality of model training

    Args:
        path (str): path to the root directory with images and csv file
        train (bool, optional): to separate the test and training samples.
        Defaults to False (generate training sample).

    Returns:
        DataFrameIterator
    """
    data_ages = pd.read_csv(path + 'labels.csv')
    # data_ages.drop_duplicates(inplace=True)
    datagen = ImageDataGenerator(validation_split=0.25,
                                 # vertical_flip=train,
                                 horizontal_flip=train,
                                 # rotation_range=45 if train else 0,
                                 rescale=1./255)

    data = datagen.flow_from_dataframe(
        dataframe=data_ages,
        directory=path+'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=15,
        class_mode='raw',
        subset='training' if train else 'validation',
        seed=12345
        )
    return data


def load_train(path: str) -> iter:
    """
    generating training sample from a dataset
    Args:
        path (str): path to the root directory with images and csv file

    Returns:
        DataFrameIterator
    """
    return load_data(path, train=True)


def load_test(path: str) -> iter:
    """
    generating train sample from a dataset
    Args:
        path (str): path to the root directory with images and csv file

    Returns:
        DataFrameIterator
    """
    return load_data(path, train=False)


def create_model(input_shape: tuple, lr: float = 1e-4) -> Sequential:
    """
    creating a training model

    Args:
        input_shape (tuple): shape datasets
        lr (float, optional): floating point value learning rate.
            Defaults to 1e-4.

    Returns:
        Sequential: training model
    """
    model = Sequential()

    backbone = ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False)

    # не замораживаем слои ResNet50
    backbone.trainable = True
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])
    model.summary()

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=6,
                steps_per_epoch=None, validation_steps=None):
    """_summary_

    Args:
        model (Sequential): train model
        train_data (iter): data for model training
        test_data (iter): data for model testing
        batch_size (int, optional): batch size. Defaults to None.
        epochs (int, optional): count epochs. Defaults to 5.
        steps_per_epoch (int, optional): number of steps per epoch.
            Defaults to None.
        validation_steps (int, optional): number of steps validation.
            Defaults to None.

    Returns:
        Sequential: trained model
    """
    if not steps_per_epoch:
        steps_per_epoch = len(train_data)
    if not validation_steps:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model

# for local testing
# if __name__ == "__main__":
#     train_data = load_train('/home/oslik/test_jpg/datasets/faces/')
#     test_data = load_train('/home/oslik/test_jpg/datasets/faces/')
#     model = create_model((150, 150, 3), lr=1)
#     print(
#         train_model(
#             model,
#             train_data=train_data,
#             test_data=test_data,
#             epochs=1
#             )
#         )



"""
2023-08-12 17:54:53.708784: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6

2023-08-12 17:54:53.710524: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6

2023-08-12 17:54:54.556839: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1

2023-08-12 17:54:55.228006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 

pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0

coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s

2023-08-12 17:54:55.228083: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1

2023-08-12 17:54:55.228115: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10

2023-08-12 17:54:55.230233: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10

2023-08-12 17:54:55.230743: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10

2023-08-12 17:54:55.232847: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10

2023-08-12 17:54:55.233986: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10

2023-08-12 17:54:55.234052: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7

2023-08-12 17:54:55.238418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0

Using TensorFlow backend.

Found 5694 validated image filenames.

Found 1897 validated image filenames.

2023-08-12 17:54:55.413542: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA

2023-08-12 17:54:55.419837: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2099995000 Hz

2023-08-12 17:54:55.420342: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4368cb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

2023-08-12 17:54:55.420371: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

2023-08-12 17:54:55.565706: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x39c8fd0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:

2023-08-12 17:54:55.565742: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0

2023-08-12 17:54:55.568115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 

pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0

coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s

2023-08-12 17:54:55.568174: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1

2023-08-12 17:54:55.568185: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10

2023-08-12 17:54:55.568211: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10

2023-08-12 17:54:55.568220: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10

2023-08-12 17:54:55.568230: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10

2023-08-12 17:54:55.568239: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10

2023-08-12 17:54:55.568246: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7

2023-08-12 17:54:55.572471: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0

2023-08-12 17:54:55.572542: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1

2023-08-12 17:54:55.884464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:

2023-08-12 17:54:55.884517: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 

2023-08-12 17:54:55.884525: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 

2023-08-12 17:54:55.888980: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.

2023-08-12 17:54:55.889028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10240 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8b:00.0, compute capability: 7.0)

Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5


    8192/94765736 [..............................] - ETA: 2s
  237568/94765736 [..............................] - ETA: 20s
 1236992/94765736 [..............................] - ETA: 7s 
 6750208/94765736 [=>............................] - ETA: 1s
 8650752/94765736 [=>............................] - ETA: 1s
12697600/94765736 [===>..........................] - ETA: 1s
16941056/94765736 [====>.........................] - ETA: 1s
21176320/94765736 [=====>........................] - ETA: 1s
25403392/94765736 [=======>......................] - ETA: 1s
29655040/94765736 [========>.....................] - ETA: 1s
33882112/94765736 [=========>....................] - ETA: 1s
38117376/94765736 [===========>..................] - ETA: 1s
42352640/94765736 [============>.................] - ETA: 1s
46596096/94765736 [=============>................] - ETA: 1s
50839552/94765736 [===============>..............] - ETA: 1s
55066624/94765736 [================>.............] - ETA: 0s
58515456/94765736 [=================>............] - ETA: 0s
60645376/94765736 [==================>...........] - ETA: 0s
63504384/94765736 [===================>..........] - ETA: 0s
67706880/94765736 [====================>.........] - ETA: 0s
71901184/94765736 [=====================>........] - ETA: 0s
72065024/94765736 [=====================>........] - ETA: 0s
76144640/94765736 [=======================>......] - ETA: 0s
80363520/94765736 [========================>.....] - ETA: 0s
84393984/94765736 [=========================>....] - ETA: 0s
84647936/94765736 [=========================>....] - ETA: 0s
88801280/94765736 [===========================>..] - ETA: 0s
92692480/94765736 [============================>.] - ETA: 0s
93085696/94765736 [============================>.] - ETA: 0s
94773248/94765736 [==============================] - 2s 0us/step

Model: "sequential"

_________________________________________________________________

Layer (type)                 Output Shape              Param #   

=================================================================

resnet50 (Model)             (None, 7, 7, 2048)        23587712  

_________________________________________________________________

global_average_pooling2d (Gl (None, 2048)              0         

_________________________________________________________________

dense (Dense)                (None, 1)                 2049      

=================================================================

Total params: 23,589,761

Trainable params: 23,536,641

Non-trainable params: 53,120

_________________________________________________________________

<class 'tensorflow.python.keras.engine.sequential.Sequential'>

WARNING:tensorflow:sample_weight modes were coerced from

  ...

    to  

  ['...']

WARNING:tensorflow:sample_weight modes were coerced from

  ...

    to  

  ['...']

Train for 380 steps, validate for 127 steps

Epoch 1/6

2023-08-12 17:55:08.825213: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10

2023-08-12 17:55:09.108492: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7

380/380 - 48s - loss: 218.0081 - mae: 10.6287 - val_loss: 648.4815 - val_mae: 20.2946

Epoch 2/6

380/380 - 39s - loss: 85.7651 - mae: 7.1153 - val_loss: 114.7910 - val_mae: 8.1687

Epoch 3/6

380/380 - 38s - loss: 57.5429 - mae: 5.8100 - val_loss: 81.3813 - val_mae: 6.7181

Epoch 4/6

380/380 - 38s - loss: 44.8881 - mae: 5.1233 - val_loss: 77.5937 - val_mae: 6.6238

Epoch 5/6

380/380 - 39s - loss: 34.4057 - mae: 4.5054 - val_loss: 71.3283 - val_mae: 6.4673

Epoch 6/6

380/380 - 44s - loss: 25.2132 - mae: 3.8485 - val_loss: 73.1489 - val_mae: 6.2157

WARNING:tensorflow:sample_weight modes were coerced from

  ...

    to  

  ['...']

127/127 - 10s - loss: 73.1489 - mae: 6.2157

Test MAE: 6.2157
"""
