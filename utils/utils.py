# -*- coding: utf-8 -*
import os
import json
import keras
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops, math_ops


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_png(contents, name=None):
    r"""Convenience function to check if the 'contents' encodes a PNG image.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)

  Returns:
     A scalar boolean tensor indicating if 'contents' may be a PNG image.
     is_png is susceptible to false positives.
  """
    with ops.name_scope(name, 'is_png'):
        substr = string_ops.substr(contents, 0, 3)
        return math_ops.equal(substr, b'\211PN', name=name)


def is_gif(contents, name=None):
    r"""Convenience function to check if the 'contents' encodes a gif image.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)

  Returns:
     A scalar boolean tensor indicating if 'contents' may be a PNG image.
     is_png is susceptible to false positives.
  """
    with ops.name_scope(name, 'is_gif'):
        substr = string_ops.substr(contents, 0, 3)
        return math_ops.equal(substr, b'\x47\x49\x46', name='is_gif')
