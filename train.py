"""
@Author: Willian Antunes
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from model.model import unet_512
import numpy as np
import cv2
import os


def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))
    return data


def train_generator():
    data = list_dir('input/train')
    while True:
        for start in range(0, len(data), 16):  # 16 = Batch Size
            x_batch = []
            y_batch = []
            end = min(start + 16, len(data))
            id_train_batch = data[start:end]
            for id in id_train_batch:
                img = cv2.imread('input/train/{}.jpg'.format(id))
                img = cv2.resize(img, (128, 128))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (128, 128))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    data = list_dir('input/valid')
    while True:
        for start in range(0, len(data), 16):  # 16 = Batch Size
            x_batch = []
            y_batch = []
            end = min(start + 16, len(data))
            id_train_batch = data[start:end]
            for id in id_train_batch:
                img = cv2.imread('input/valid/{}.jpg'.format(id))
                img = cv2.resize(img, (128, 128))
                mask = cv2.imread('input/valid_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (128, 128))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch



callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]


data_train = list_dir('input/train')
data_valid = list_dir('input/valid')
model = unet_512()
model.fit(train_generator(), callbacks=callbacks, verbose=1, epochs=300, steps_per_epoch=np.ceil(float(len(data_train)) / float(16)), validation_data=valid_generator(), validation_steps=np.ceil(float(len(data_valid)) / float(16)))
