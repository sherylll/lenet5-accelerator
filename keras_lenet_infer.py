'''
CPU (Intel i7-7500 CPU @ 2.0GHz)
N = [10, 100, 1000, 10000]
latency = [0.0078, 0.00047, 0.000219, 0.000199], acceleration flattens out due to limited memory on a mobile cpu
GPU (GeForce 940MX)
latency = [0.2383,0.0128, 0.00132, 0.0002]
'''

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import time
import sys
saved_model_dir = 'misc/saved_model.json'
saved_weights_dir = 'misc/saved_weights.h5'

if __name__ == "__main__":
    with open(saved_model_dir) as f:
        json_str = f.read()
    model = model_from_json(json_str)
    model.load_weights(saved_weights_dir)

    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    num_samples = int(sys.argv[1])
    start = time.time()
    model.predict(x_test[:num_samples])
    end = time.time()
    print((end-start)/num_samples)