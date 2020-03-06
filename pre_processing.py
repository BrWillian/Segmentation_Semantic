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
    count = -1
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)
            count += 1

    data = list(map(lambda s: s.split('.')[0], data))

    return data, count


def pre_processing():
    data_train, n_train = list_dir(train_dir)
    data_mask, n_masks = list_dir(train_masks_dir)
    gen = simple_generator()

    if n_train != n_masks:
        print('WARNING: There are more masks or images')
        return

    data_train = sorted(data_train)
    data_mask = sorted(data_mask)

    for x,y in zip(data_train, data_mask):
        id = next(gen)
        os.rename(train_dir + x + '.jpg', train_dir + 'img_{}.jpg'.format(id))
        os.rename(train_masks_dir + y + '.png', train_masks_dir + 'img_{}_mask.png'.format(id))


if __name__ == '__main__':
    pre_processing()