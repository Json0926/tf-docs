#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : dataAugmentation.py
#   Author      : Js
#   Created date: 2020/4/1 下午5:50
#   Description :
#
#================================================================

import urllib

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_datasets as tfds
import PIL.Image

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)

import numpy as np

image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
PIL.Image.open(image_path)

image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string, channels=3)

#
# def visualize(original, augmented):
#     fig = plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.title('Original image')
#     plt.imshow(original)
#
#     plt.subplot(1, 2, 2)
#     plt.title('Augmented image')
#     plt.imshow(augmented)
#
#
# flipped = tf.image.flip_left_right(image)
# visualize(image, flipped)

dataset, info = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples = info.splits['train'].num_examples


def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def augment(image, label):
    image, label = convert(image, label)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 34, 34)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image, label

BATCH_SIZE = 64
NUM_EXAMPLES = 2048

augment_train_batches = (
    train_dataset.take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

non_augmented_train_batched = (
    train_dataset.take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_batches = (
    test_dataset
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(2*BATCH_SIZE)
)


def make_model():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model_without_aug = make_model()
no_aug_history = model_without_aug.fit(non_augmented_train_batched, epochs=50,
                                       validation_data=validation_batches)

model_with_aug = make_model()
aug_history = model_with_aug.fit(augment_train_batches, epochs=50,
                                 validation_data=validation_batches)

plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Augmented': aug_history, 'Non-Augmented': no_aug_history}, metric='accuracy')
plt.title('Accuracy')
plt.ylim([0.75, 1])
plt.show()
