'''
Saves mnist images to txt files
'''

import numpy as np
import keras

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 255

# print('shape shape:', x_test.shape)
# print(, 'test samples')
x_test_len = x_test.shape[0]

for i in range(x_test_len):
    with open('test_images/'+str(i)+'.txt', 'w') as outfile:
        outfile.write('# '+str(y_test[i])+'\n')
np.savetxt(outfile, x_test[i])