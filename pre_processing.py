"""
@Autor Willian Antunes
"""
import sys
import os


def simple_generator():
    i = 25000
    while True:
        i += 1
        yield i


def dirs():
    train_dir = sys.argv[2] + 'train/'
    train_masks_dir = sys.argv[2] + 'train_masks/'
    valid_dir = sys.argv[2] + 'valid/'
    valid_mask_dir = sys.argv[2] + 'valid_masks/'

    return train_dir, train_masks_dir, valid_dir, valid_mask_dir

def verify_dir():
    if not os.path.isdir(sys.argv[2]):
        print('Directory is not valid use --dir and define directory from dataset')


def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))
    data = list(map(lambda s: s.split('_')[-1], data))

    return data


def pre_processing():
    train_dir, train_masks_dir, valid_dir, valid_masks_dir = dirs()
    data_train = list_dir(train_dir)
    data_mask = list_dir(train_masks_dir)
    valid = list_dir(valid_dir)
    valid_mask = list_dir(valid_masks_dir)
    gen_train = simple_generator()
    gen_valid = simple_generator()


    data_train = list(map(lambda s: s.split('img_')[-1], data_train))
    data_train = [int(val) for val in data_train]
    data_train = sorted(data_train)

    data_mask = list(map(lambda s: s.split('_mask')[0], data_mask))
    data_mask = list(map(lambda s: s.split('img_')[-1], data_mask))
    data_mask = [int(val) for val in data_mask]
    data_mask = sorted(data_mask)
    #print(data_mask)

    valid = sorted(valid)
    valid_mask = sorted(valid_mask)


    for x, y in zip(data_train, data_mask):
        id = next(gen_train)
        os.rename(train_dir + 'img_' + str(x) + '.jpg', train_dir + '{}.jpg'.format(id))
        os.rename(train_masks_dir + 'img_' + str(y) + '_mask.png', train_masks_dir + '{}_mask.png'.format(id))

    #for x, y in zip(valid, valid_mask):
    #    id = next(gen_valid)
    #    os.rename(valid_dir + x + '.png', valid_dir + 'img_0{}.png'.format(id))
    #    os.rename(valid_masks_dir + y + '.png', valid_masks_dir + 'img_0{}_mask.png'.format(id))


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage python3 pre-processing.py --dir [Dataset path (train, train_masks, valid, valid_masks]')
        exit()
    elif not verify_dir():
        pre_processing()