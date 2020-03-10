import cv2
import os
import sys
from matplotlib import pyplot as plt
from pre_processing import list_dir, simple_generator


def dirs():
    train_dir = sys.argv[2]
    train_masks_dir = sys.argv[4]

    return train_dir, train_masks_dir


def pre_processing():
    train_dir, train_masks_dir = dirs()
    data_train = list_dir(train_dir)
    data_mask = list_dir(train_masks_dir)
    gen_train = simple_generator()


    data_train = sorted(data_train)
    data_mask = sorted(data_mask)


    for x, y in zip(data_train, data_mask):
        id = next(gen_train)
        os.rename(train_dir + x + '.jpg', train_dir + 'img_{}.jpg'.format(id))
        os.rename(train_masks_dir + y + '.png', train_masks_dir + 'img_{}_mask.png'.format(id))


def make_masks():
    images = []
    masks_dir = list_dir('/home/willian/Downloads/masks/')
    for x in masks_dir:
        img = '/home/willian/Downloads/masks/'+x+'.png'
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        cropped_image = img[47:417,]
        _, img = cv2.threshold(cropped_image, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite('/home/willian/Downloads/masks/'+x+'_mask.png', img)
