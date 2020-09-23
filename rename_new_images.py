import os
import sys



def gen():
    i = 3000000
    while(True):
        i += 1
        yield i


def list_dir(dir):
    data = []
    for _, _, files in os.walk(dir+'imgs/'):
        for file in files:
            data.append(file)

    data = list(map(lambda s: s.split('.')[0], data))
    data = list(map(lambda s: s.split('img_')[-1], data))
    data = [int(val) for val in data]
    print(data)
    return data

def rename(dir):
    images = list_dir(dir)
    images = sorted(images)
    generator = gen()

    for img in images:
        id = next(generator)
        os.rename(dir + 'imgs/' + str(img) + '.jpg', dir + 'imgs/' + '{}.jpg'.format(id))
        os.rename(dir + 'masks/' + str(img) + '_mask.png', dir + 'masks/' + '{}_mask.png'.format(id))


if __name__ == '__main__':


    if len(sys.argv) < 2:
        print('Usage script type python3 rename_new_images.py --dir [dir of images]')
    else:
        rename(sys.argv[2])


