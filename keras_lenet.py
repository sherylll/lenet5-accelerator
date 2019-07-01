import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
saved_model_dir = 'saved_model.json'
saved_weights_dir = 'saved_weights.h5'
batch_size = 128
num_classes = 10
epochs = 10

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

inputs = Input(shape=input_shape)
layer = Conv2D(filters=6,
               kernel_size=5,
               strides=1,
               activation='relu',
               input_shape=input_shape)(inputs)

# Pooling layer 1
layer = MaxPooling2D(pool_size=2, strides=2)(layer)

# Layer 2
# Conv Layer 2
layer = Conv2D(filters=16,
               kernel_size=5,
               strides=1,
               activation='relu',
               input_shape=(12, 12, 6))(layer)
# Pooling Layer 2
layer = MaxPooling2D(pool_size=2, strides=2)(layer)
# Flatten
layer = Flatten()(layer)
# Layer 3
# Fully connected layer 1
layer = Dense(units=120, activation='relu')(layer)
# Layer 4
# Fully connected layer 2
layer = Dense(units=84, activation='relu')(layer)
# Layer 5
# Output Layer
predictions = Dense(units=10, activation='softmax')(layer)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

json_string = model.to_json()
with open(saved_model_dir, "w+") as f:
    f.write(json_string)
model.save_weights(saved_weights_dir)
