# reference : https://www.tensorflow.org/tutorials/keras/regression?hl=zh-tw

from asyncio.windows_events import NULL
from os import sep
from pickletools import optimize
from tabnanny import verbose
from time import time
from turtle import clear
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import Normalizer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras import layers
np.set_printoptions(precision = 3, suppress = True)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names = column_names, na_values = '?', comment = '\t',
                          sep = ' ', skipinitialspace = True)

dataset = raw_dataset.copy()
# print(dataset.head())
# print(dataset.tail())

# isna() is same with isnull()
# print(dataset.isna().sum)

# drop the rows which have any N/A
dataset = dataset.dropna()

# map 1 to 'USA', 2 to 'Europe', and 3 to 'Japan'
dataset['Origin'] = dataset['Origin'].map({1:'USA', 2:'Europe', 3:'Japan'})
# print(dataset.tail())

# achieve 'one hoe encode', split column 'Origin' to 'USA', 'Europe', and 'Japan'
dataset = pd.get_dummies(dataset, columns = ['Origin'], prefix = '', prefix_sep = '')
# print(dataset.tail())

# split the dataset into training set and test set
# samole() has arguments
# 1) n, frac : either of the two arguments is used. Take n data or frac(0~1)*10% of the data is taken.
# 2) random_state : something like random seed.
# 3) replace : determine whether the data will be put back after being taken. False(default) for not put back and True for put back.
#    但預設 sample 前後的資料是一樣的?!
# 4) weights : the weight of the data which is taken.
# 5) axis : None(default). 0 for indax and 1 for columns.
# print(dataset.tail())
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
# print(dataset.tail())
test_dataset = dataset.drop(train_dataset.index)

# inspect the data
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

# split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_label = train_features.pop('MPG')
test_label = test_features.pop('MPG')
# print(train_label.head())

# print(train_dataset.describe().transpose()[['mean', 'std']])

# Normalization layer
normalizer = tf.keras.layers.Normalization(axis = -1)
# fit the state of the preprocessing layer to the data
normalizer.adapt(np.array(train_features))
# Calculate the mean and variance
# print(normalizer.mean.numpy())

# first = np.array(train_features[:1])
# with np.printoptions(precision = 2, suppress = True):
#     print('First example: ', first)
#     print()
#     print('Normalized: ', normalizer(first).numpy())

# print(train_features.head())

horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape = [1,], axis = None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(1)
])
# print(horsepower_model.summary())

# the model has not been trained
print(horsepower_model.predict(horsepower[:10]))

# compile the model
horsepower_model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1),
                         loss = 'mean_absolute_error')

history = horsepower_model.fit(train_features['Horsepower'], train_label, epochs = 100, verbose = 1,validation_split = 0.2)

# print(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
# print(hist.tail())

def plot_loss(history):
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_loss(history)

# store the loss
test_result = {}
test_result['horsepower_model'] = horsepower_model.evaluate(test_features['Horsepower'], test_label)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_label, label = 'Data')
    plt.plot(x, y, color = 'k', label = 'Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()
# plot_horsepower(x, y)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(1)
])

linear_model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1), loss = 'mean_absolute_error')

history = linear_model.fit(train_features, train_label, epochs = 100, validation_split = 0.2)
# plot_loss(history)

test_result['linear_model'] = linear_model.evaluate(test_features, test_label)
# print(test_result)

# DEEP NEURAL NETWORK(with hidden layers)
# use function because there are two kinds of normalization layers
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                  loss = 'mean_absolute_error')
    
    return model

# 1. single input
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
# print(dnn_horsepower_model.summary())
history = dnn_horsepower_model.fit(train_features['Horsepower'], train_label, epochs = 100, validation_split = 0.2)
# plot_loss(history)

x = tf.linspace(0.0, 250.0, 251)
y = dnn_horsepower_model.predict(x)
# plot_horsepower(x, y)

test_result['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'], test_label)


# 2. multiple inpt
dnn_model = build_and_compile_model(normalizer)
# print(dnn_model.summary())

history = dnn_model.fit(train_features, train_label, epochs = 100, validation_split = 0.2)
# plot_loss(history)
test_result['dnn_model'] = dnn_model.evaluate(test_features, test_label)

# .T is to transpose the matrix
print(pd.DataFrame(test_result, index = ['Mean absolute error [MPG]']).T)

test_predictions = dnn_model.predict(test_features).flatten()
# a = plt.axes(aspect = 'equal')
# plt.scatter(test_label, test_predictions)
# plt.xlabel('True values [MPG]')
# plt.ylabel('Predictions [MPG]')
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.plot(lims, lims)
# plt.show()

error = test_predictions - test_label
# plt.hist(error, bins = 25)
# plt.xlabel('Prediction Error [MPG]')
# plt.ylabel('Count')
# plt.show()

# if we think the model is ok, we can save it
dnn_model.save('dnn_model')

# to reload the model
reloaded = tf.keras.models.load_model('dnn_model')

test_result['reloaded'] = reloaded.evaluate(test_features, test_label)
print(pd.DataFrame(test_result, index=['Mean absolute error [MPG]']).T)