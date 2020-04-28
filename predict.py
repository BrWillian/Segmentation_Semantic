"""
@Author Willian Antunes
"""
import os
import cv2
import sys
import numpy as np
from model.model import unet_256


def load_image_for_predict(img, input_size=(256, 256)):
    data = []
    data.append(img)
    img_name = list(map(lambda s: s.split('/')[-1], data))
    img_name = list(map(lambda s: s.split('.')[0], img_name))
    img = cv2.imread(img)
    img = cv2.resize(img, (625, 352))
    img = img[47:336, ]
    img = cv2.resize(img, input_size)
    img = img.reshape((1,) + img.shape)
    img = np.array(img, np.float32) / 255

    return img, img_name


def predict(img, output_size=(512, 256)):

    model = unet_256()
    model.load_weights(filepath='weights/best_weights_8.hdf5')

    x_batch, img_name = load_image_for_predict(img)

    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    preds = np.array(preds, np.float32) * 100
    preds = preds.transpose(1, 0, 2).reshape(-1, preds.shape[1])

    preds = cv2.resize(preds, output_size)

    p1 = preds[:, 0:172]

    p2 = preds[:, 172:342]

    p3 = preds[:, 342:512]

    cv2.imwrite('predict/new_predict/' + str(img_name[0]) + '_predict.png', preds)
    #cv2.imwrite('predict/' + str(img_name[0]) + '_predict_left.png', p1)
    #cv2.imwrite('predict/' + str(img_name[0]) + '_predict_center.png', p2)
    #cv2.imwrite('predict/' + str(img_name[0]) + '_predict_right.png', p3)








if __name__ == "__main__":
    predict('predict/new_predict/moto3.jpeg')



