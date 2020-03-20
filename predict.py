"""
@Author Willian Antunes
"""
import os
import cv2
import sys
import numpy as np
from model.model import unet_256

def load_image_for_predict(img, input_size=(256, 256)):
    img = cv2.imread(img)
    img = cv2.resize(img, (256, 256))
    img = img.reshape((1,) + img.shape)
    img = np.array(img, np.float32) / 255

    print(img.shape)
    return img


def predict(img):
    model = unet_256()
    model.load_weights(filepath='weights/best_weights_4.hdf5')

    x_batch = load_image_for_predict(img)
    print(x_batch.shape)

    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    preds = preds.transpose(1, 0, 2).reshape(-1, preds.shape[1])

    preds = cv2.resize(preds, (512, 256))
    cv2.imwrite('novo.png', preds)




if __name__ == "__main__":
    predict('input/valid/192.jpg')