from PIL import Image

import numpy as np
import os
import glob

import tensorflow as tf

'''
Checks all images from the AVA dataset if they have corrupted jpegs, and lists them for removal.

Removal must be done manually !
'''

base_images_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/images/images/'
ava_dataset_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/AVA.txt'

IMAGE_SIZE = 128
BASE_LEN = len(base_images_path) - 1

files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)

        count = 255000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %0.2f of the dataset' % (i / 255000. * 100))

train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')


def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    return image


def is_jpg(filename):
    try:
        ii = Image.open(filename)
        return ii.format == 'JPEG'
    except IOError:
        return False


def is_valid_jpg(jpg_file):
    if jpg_file.split('.')[-1].lower() == 'jpg':
        with open(jpg_file, 'rb') as ff:
            ff.seek(-2, 2)
            return ff.read() == '\xff\xd9'
    else:
        return True


sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())

    count = 0
    fn = tf.placeholder(dtype=tf.string)
    img = parse_data(fn)

    for path in train_image_paths:
        try:
            sess.run(img, feed_dict={fn: path})
        except Exception as e:
            os.rename(path, path + '_bk')
            print(path, "failed to load !")
            print()
            count += 1
            continue
        print(count, "images load !")
        try:
            if is_jpg(path):
                if not is_valid_jpg(path):
                    os.rename(path, path + '_bk')
                    count += 1
                    continue
        except Exception as e:
            os.rename(path, path + '_bk')
            print(path, "failed to load !")
            print()
            count += 1
            continue
    print(count, "images failed to load !")

print("All done !")
