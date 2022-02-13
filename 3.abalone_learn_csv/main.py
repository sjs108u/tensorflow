# reference : https://www.tensorflow.org/tutorials/load_data/csv?hl=zh-tw

import pandas as pd
import numpy as np
import tensorflow as tf
import urllib
import os

np.set_printoptions(precision = 3, suppress = True)

if not os.path.exists('abalone_train.csv'):
    content = urllib.request.urlopen('https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv')
    with open('abalone_train.csv', 'w') as f:
        f.write(content.read().decode('utf-8'))

abalone_train = pd.read_csv(
    'abalone_train.csv',
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

# print(abalone_train.head())

abalone_feature = abalone_train.copy()
abalone_label = abalone_feature.pop('Age')

# print(abalone_feature)
# print(abalone_label)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam',
              loss = tf.losses.MeanSquaredError())

model.fit(abalone_feature, abalone_label, epochs = 10)