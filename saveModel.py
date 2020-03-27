#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : saveModel.py
#   Author      : Js
#   Created date: 2020/3/27 下午5:04
#   Description :
#
#================================================================

import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# model = create_model()
# model.summary()


'''
创建一个保存模型权重的回调 
'''
# checkpoint_path = 'training_1/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# 新模型
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print('Untrained model accuracy: {:5.2f}%'.format(100*acc))
#
# # load weight
# model.load_weights(checkpoint_path)
# lossL, accL = model.evaluate(test_images, test_labels, verbose=2)
# print('restored model accuracy: {:5.2f}%'.format(100*accL))

'''
save model option
'''
# checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  verbose=1,
#                                                  save_weights_only=True,
#                                                  period=5)

# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit(train_images, train_labels, epochs=50,
#           callbacks=[cp_callback], validation_data=(test_images, test_labels),
#           verbose=0)


'''
save whole model
'''
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

new_model = keras.models.load_model('my_model.h5')
new_model.summary()


'''
save model by saved_model
'''
