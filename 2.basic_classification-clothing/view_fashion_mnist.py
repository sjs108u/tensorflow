from msilib.schema import MsiPatchHeaders
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

for idx in range(10):
    train_image = train_images[idx]

    for i in train_image:
        for j in i:
            if j > 0:
                print('*', end = '')
            else:
                print(' ', end = '')
        print('')