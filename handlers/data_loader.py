# -*- coding: utf-8 -*
import glob
import os

import numpy as np
from pandas import json


def load_tid_data(image_dir, score_dir):
    image_paths = []
    image_scores = []
    files = glob.glob(image_dir + "*.bmp")
    files = sorted(files)
    f = open(score_dir, 'r')
    data = json.load(f)
    for item in data:
        image_id = item['image_id']
        label = item['label']
        file_path = image_dir + image_id + '.bmp'
        file_path_bmp = score_dir + image_id + '.BMP'
        if os.path.exists(file_path):
            image_paths.append(file_path)
            image_scores.append(label)
        elif os.path.exists(file_path_bmp):
            image_paths.append(file_path_bmp)
            image_scores.append(label)

    image_paths = np.array(image_paths)
    image_scores = np.array(image_scores, dtype='float32')
    print('load_tid_data set size : ', image_paths.shape, image_scores.shape)
    print('Train and validation datasets ready !')
    return image_paths, image_scores


def load_ava_data(image_dir, score_dir):
    image_paths = []
    image_scores = []
    files = glob.glob(image_dir + "*.jpg")
    files = sorted(files)
    f = open(score_dir, 'r')
    lines = f.readlines()
    image_size = 0
    for i, line in enumerate(lines):
        token = line.split()
        image_id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = image_dir + str(image_id) + '.jpg'
        if os.path.exists(file_path):
            image_paths.append(file_path)
            image_scores.append(values)
            image_size = image_size + 1

    print('Loaded Finish')
    image_paths = np.array(image_paths)
    image_scores = np.array(image_scores, dtype='float32')
    print('load_ava_data set size : ', image_paths.shape, image_scores.shape)
    print('Train and validation datasets ready !')
    return image_paths, image_scores
