#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : text_classification_with_hub.py
#   Author      : Js
#   Created date: 2020/3/26 下午4:30
#   Description :
#
#================================================================

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(
    name='imdb_reviews',
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=False
)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

