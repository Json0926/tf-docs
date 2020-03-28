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

import tensorflow as tf
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)
