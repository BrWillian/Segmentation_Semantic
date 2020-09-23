"""
@Autor Willian Antunes
"""
import cv2
from pre_processing import list_dir, simple_generator
import numpy as np


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomShiftScaleRotate(image, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45), aspect_limit=(0, 0), borderMode=cv2.BORDER_CONSTANT, u=0.5):

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def aug_generator():
    data = list_dir('input/train')
    data = [int(val) for val in data]
    data = sorted(data)
    gen = simple_generator()
    x = 2
    while x:
        for start in range(0, len(data), 1):  # 16 = Batch Size
            end = min(start + 1, len(data))
            id_train_batch = data[start:end]
            for id in id_train_batch:
                img = cv2.imread('input/train/{}.jpg'.format(id))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                #img = cv2.resize(img, (625, 399))
                #mask = cv2.resize(mask, (625, 353))
                #img = img[26:384, ]
                #mask = mask[6:336, ]


                #img = cv2.resize(img, (512, 256))
                #mask = cv2.resize(mask, (512, 256))
                mask = np.expand_dims(mask, axis=2)
                img, mask = randomShiftScaleRotate(img, mask, shift_limit=(
                np.random.uniform(-0.0625, -0.0725), np.random.uniform(0.0625, 0.0725)), scale_limit=(
                np.random.uniform(-0.1, -0.5), np.random.uniform(0.1, 0.5)), rotate_limit=(-0, 0))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(np.random.uniform(-25, -100), np.random.uniform(25, 100)),
                                               sat_shift_limit=(np.random.uniform(-45, 0), np.random.uniform(0, 45)),
                                               val_shift_limit=(np.random.uniform(-50, 0), np.random.uniform(0, 50)))
                img, mask = randomHorizontalFlip(img, mask)
                id_x = next(gen)
                cv2.imwrite('input/aug/{}.jpg'.format(id_x), img)
                cv2.imwrite('input/aug_masks/{}_mask.png'.format(id_x), mask)
                break
        x = x - 1







if __name__ == "__main__":
    aug_generator()