#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : estimator.py
#   Author      : Js
#   Created date: 2020/3/30 下午3:47
#   Description :
#
#================================================================

import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    'iris_training.csv',
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    'iris_test.csv',
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Species')
test_y = test.pop('Species')

'''
创建输入函数
'''
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


'''
定义特征列
'''
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


'''
实例化Estimator
'''
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3)


'''
训练模型
'''
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),
                 steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('Test set acc: {accuracy: 0.3f}'.format(**eval_result))


'''
用模型预测
'''
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fnp(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(input_fn=lambda: input_fnp(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is {} ({:.1f}%), expected {}'.format(
        SPECIES[class_id], 100 * probability, expec
    ))
