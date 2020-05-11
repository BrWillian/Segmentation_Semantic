import cv2
import os
import sys
from itertools import product
from pre_processing import list_dir, simple_generator


def dirs():
    train_dir = sys.argv[2]
    train_masks_dir = sys.argv[4]

    return train_dir, train_masks_dir


def pre_processing():
    train_dir, train_masks_dir = dirs()
    train_masks_dir = '/home/willian/Downloads/masks/'
    #data_train = list_dir(train_dir)
    data_mask = list_dir('/home/willian/Downloads/masks/')
    gen_train = simple_generator()


    #data_train = sorted(data_train)
    data_mask = sorted(data_mask)


    for x in data_mask:
        id = next(gen_train)
        os.rename(train_dir + x + '.jpg', train_dir + 'img_{}.jpg'.format(id))
        os.rename(train_masks_dir + x + '.png', train_masks_dir + 'img_{}_mask.png'.format(id))


def make_masks():
    #images = []
    masks_dir = list_dir('/home/willian/Downloads/masks/')
    for x in masks_dir:
        img = '/home/willian/Downloads/masks/'+x+'.png'
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        cropped_image = img[26:384, ]
        _, img = cv2.threshold(cropped_image, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite('/home/willian/Downloads/masks/'+x+'.png', img)


def list_dir_1(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))
    data = list(map(lambda s: s.split('-')[0], data))

    return data

def delete():
    masks = list_dir_1('/home/willian/Downloads/masks/')

    images = list_dir_1('/home/willian/Downloads/imgs/')

    masks = sorted(masks)
    images = sorted(images)

    print(masks)
    #print(images)

    for i in images:
        if i not in masks:
            os.remove('/home/willian/Downloads/imgs/'+i+'.jpg')

make_masks()