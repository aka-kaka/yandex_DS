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
        target_size=(150, 150),
        batch_size=16,  # 32
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

    # не замораживаем верхушку ResNet50
    backbone.trainable = True
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])
    model.summary()

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=8,
                steps_per_epoch=None, validation_steps=None):
    """_summary_

    Args:
        model (Sequential): train model
        train_data (iter): data for model training
        test_data (iter): data for model testing
        batch_size (int, optional): batch size. Defaults to None.
        epochs (int, optional): count epochs. Defaults to 14.
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


if __name__ == "__main__":
    train_data = load_train('/home/oslik/test_jpg/datasets/faces/')
    test_data = load_train('/home/oslik/test_jpg/datasets/faces/')
    len(test_data)
#    model = create_model((150, 150, 3), lr=1)
#    print(
#        train_model(
#            model,
#            train_data=train_data,
#            test_data=test_data,
#            epochs=1
#            )
#        )