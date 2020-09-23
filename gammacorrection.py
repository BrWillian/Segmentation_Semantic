from __future__ import print_function
import numpy as np
import cv2
import os


class TransformImages:
    def __init__(self, directory):
        self.dir = directory
        self.imgs = []
        self.list_dir()

    def list_dir(self):
        self.imgs = []
        for root, paths, files in os.walk(self.dir):
            for file in files:
                self.imgs.append(file)

    def adjust(self):
        for im in self.imgs:
            img = cv2.imread(self.dir + im, cv2.IMREAD_COLOR)
            img_adjusted = self.adjust_gamma(img, gamma=1.6)
            img_adjusted = self.increase_brightness(img_adjusted, value=7)
            cv2.imwrite('result/' + im, img_adjusted)

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def increase_brightness(image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def suffle(self, directory, n=20):
        self.imgs = []
        for root, paths, files in os.walk(directory):
            for file in files:
                self.imgs.append(file)

        index = np.random.randint(0, len(self.imgs), n)

        list_imgs = [self.imgs[i] for i in index]

        for img in list_imgs:
            try:
                os.rename(directory+img, './input/suffle/train/'+img)
                os.rename('./input/train_masks/'+img.split('.')[0]+'_mask.png', './input/suffle/train_masks/'+img.split('.')[0]+'_mask.png')
            except Exception as e:
                print(str(e))


obj = TransformImages('/home/willian/images/noite/')
obj.adjust()