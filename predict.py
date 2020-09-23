"""
@Author Willian Antunes
"""
import os
import cv2
import sys
import numpy as np
from model.model import unet_256
import tensorflow as tf
import segmentation_models as sm

model = sm.Unet('vgg16', encoder_weights='imagenet')

model.load_weights(filepath='weights/best_weights_1.hdf5')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(model.summary())


def load_image_for_predict(img, input_size=(256, 256)):
    data = []
    data.append(img)
    img_name = list(map(lambda s: s.split('/')[-1], data))
    img_name = list(map(lambda s: s.split('.')[0], img_name))
    img = cv2.imread(img)
    #img = cv2.resize(img, (625, 352))
    img = img[:405, ]
    img = cv2.resize(img, input_size)
    img = img.reshape((1,) + img.shape)
    img = np.array(img, np.float32) / 255

    return img, img_name


def predict(img, output_size=(512, 256)):
    x_batch, img_name = load_image_for_predict(img)

    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    preds = np.array(preds, np.float32) * 255
    preds = preds.transpose(1, 0, 2).reshape(-1, preds.shape[1])

    preds = cv2.resize(preds, output_size)

    cv2.imwrite('/home/PycharmProjects/clientapi/73_569/' + str(img_name[0]) + '_predict.png', preds)
    # cv2.imwrite('predict/' + str(img_name[0]) + '_predict_left.png', p1)
    # cv2.imwrite('predict/' + str(img_name[0]) + '_predict_center.png', p2)
    # cv2.imwrite('predict/' + str(img_name[0]) + '_predict_right.png', p3)


if __name__ == "__main__":

    # predict('/home/willian/image.jpg')
    for root, paths, files in os.walk('/home/PycharmProjects/clientapi/73_569/'):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                print(path)
                predict(path)
