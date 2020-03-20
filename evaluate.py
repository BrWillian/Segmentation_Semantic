import os
import cv2
import numpy as np
from model.model import unet_256

model = unet_256()

test_dir = 'input/test'

def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))
    return data


def test_generator():
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

def evaluate():
    model.load_weights(filepath='weights/best_weights_1.hdf5')
    model.evaluate_generator(test_generator(), verbose=1)


evaluate()