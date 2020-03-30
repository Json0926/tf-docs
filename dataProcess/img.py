#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : img.py
#   Author      : Js
#   Created date: 2020/3/28 下午8:57
#   Description :
#
#================================================================

import os
import tensorflow as tf
import pathlib
import random
import IPython

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=False)
data_root = pathlib.Path(data_root_orig)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

attributions = (data_root/'LICENSE.txt').open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return 'Image (CC BY 2.0)' + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
''' img path '''
img_path = all_image_paths[0]
''' raw img '''
img_raw = tf.io.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)

img_final = tf.image.resize(img_tensor, [192, 192])
img_final - img_final / 255.0


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image = image/255.0

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
plt.show()
