#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : overfit.py
#   Author      : Js
#   Created date: 2020/3/27 下午2:07
#   Description :
#
#================================================================

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

# import tensorflow_docs as tfdocs

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/'tensorboard_logs'
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz',
                             'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')

