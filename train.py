"""
@Author: Willian Antunes
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from model.model import unet_128, unet_256
import numpy as np
import cv2
import os
from tensorflow.keras.

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
                img = cv2.resize(img, (256, 256))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (256, 256))
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
                img = cv2.resize(img, (256, 256))
                mask = cv2.imread('input/valid_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (256, 256))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def verify_model(n=0):
    while os.path.isfile('weights/best_weights_{}.hdf5'.format(n)):
        n += 1
    return 'weights/best_weights_{}.hdf5'.format(n)


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath=verify_model(),
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1),
             TensorBoard(log_dir='logs')]


data_train = list_dir('input/train')
data_valid = list_dir('input/valid')
model = unet_256()
model.fit(train_generator(), callbacks=callbacks, verbose=1, epochs=30, steps_per_epoch=np.ceil(float(len(data_train)) / float(16)), validation_data=valid_generator(), validation_steps=np.ceil(float(len(data_valid)) / float(16)))
