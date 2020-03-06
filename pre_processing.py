"""
@Autor Willian Antunes
"""
import sys
import os

train_dir = sys.argv[2] + 'train/'
train_masks_dir = sys.argv[2] + 'train_masks/'
valid_dir = sys.argv[2] + 'valid/'
valid_mask_dir = sys.argv[2] + 'valid_masks/'


def simple_generator():
    i = 0
    while True:
        i += 1
        yield i


def verify_dir():
    if not os.path.isdir(sys.argv[2]):
        print('Directory is not valid use --dir and define directory from dataset')


def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))

    return data


def pre_processing():
    data_train = list_dir(train_dir)
    data_mask = list_dir(train_masks_dir)
    valid = list_dir(valid_dir)
    valid_mask = list_dir(valid_mask_dir)
    gen_train = simple_generator()
    gen_valid = simple_generator()

    data_train = sorted(data_train)
    data_mask = sorted(data_mask)

    valid = sorted(valid)
    valid_mask = sorted(valid_mask)

    for x, y in zip(data_train, data_mask):
        id = next(gen_train)
        os.rename(train_dir + x + '.jpg', train_dir + 'img_{}.jpg'.format(id))
        os.rename(train_masks_dir + y + '.png', train_masks_dir + 'img_{}_mask.png'.format(id))

    for x, y in zip(valid, valid_mask):
        id = next(gen_valid)
        os.rename(valid_dir + x + '.jpg', valid_dir + 'img_{}.jpg'.format(id))
        os.rename(valid_mask_dir + y + '.png', valid_mask_dir + 'img_{}_mask.png'.format(id))


if __name__ == '__main__':
    pre_processing()
