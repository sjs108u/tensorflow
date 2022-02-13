# reference : https://www.tensorflow.org/tutorials/keras/text_classification?hl=zh-tw

from base64 import standard_b64decode
from cProfile import label
from doctest import Example
from numpy import vectorize
from sklearn import metrics
from sqlalchemy import true
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import shutil
import string

from tensorflow.keras import layers
from tensorflow.keras import losses

# if 'aclImdb' hasn't been downloaded
if not os.path.exists('aclImdb'):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar = True, cache_dir = '.',
                                    cache_subdir = '')
# 'aclImdb' has been downloaded 
else:
    dataset = 'aclImdb'

# concatenate 2 argments
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
# print(os.listdir(dataset_dir))

# concatenate 2 argments for training data
train_dir = os.path.join(dataset_dir, 'train')
# print(os.listdir(train_dir))

# # example for pos(itive)
# sample_file = os.path.join(train_dir, 'pos/0_9.txt')
# with open(sample_file) as f:
#     print(f.read())

# remove 'unsup'
remove_dir = os.path.join(train_dir, 'unsup')
if os.path.exists(remove_dir):
    shutil.rmtree(remove_dir)

batch_size = 32
seed = 42
# use for training
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'training',
    seed = seed # fixed number for random seed
)

# # j[0] for sentence and j[1] for labels
# for j in raw_train_ds.take(1):
#     for i in range(3):
#         print(j[0].numpy()[i])

# for text_batch, label_batch in raw_train_ds.take(1):
#     for i in range(3):
#         print('Review', text_batch.numpy()[i])
#         print('Label', label_batch.numpy()[i])

# # class_names is the name of 2 directories in 'aclImdb/train'
# print(raw_train_ds.class_names[0])
# print(raw_train_ds.class_names[1])

# use for validation
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'validation',
    seed = seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size = batch_size
)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

# this layer is to standardize, tokenize, and vectorize our data
# set the output_mode to int to create unique integer indices for each token.
vectorize_layer = layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
)

# call adapt to fit the state of the preprocessing layer to the dataset
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# # retrieve a batch (of 32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# # each token has been replaced by an integer
# print('1287 -> ', vectorize_layer.get_vocabulary()[1287])
# print('123 -> ', vectorize_layer.get_vocabulary()[123])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# apply the TextVectorization layer we created earlier to the train, validation, and test dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# .cache() keeps data in memory after it's loaded off disk.
# This will ensure the dataset does not become a bottleneck while training your model.
# If your dataset is too large to fit into memory,
# you can also use this method to create a performant on-disk cache,
# which is more efficient to read than many small files.

# .prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)

# create the model
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# print(model.summary())

# compile the model
model.compile(loss = losses.BinaryCrossentropy(from_logits = True),
              optimizer = 'adam',
              metrics = tf.metrics.BinaryAccuracy(threshold = 0.0))

# train the model
history = model.fit(train_ds, validation_data = val_ds, epochs = 10)

#evaluate the model
loss, accuracy = model.evaluate(test_ds)

print(f'Loss:{loss}, accuracy:{accuracy}')

history_dict = history.history
# # history_dict.keys() -> dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
# print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# # Loss
# # 'bo' : blue dot
# plt.plot(epochs, loss, 'bo', label = 'Training loss')
# # 'b' : blue line
# plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.legend()
# plt.show()

# # Accuracy
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')

# plt.legend(loc='lower right')
# plt.show()

# Export the model
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(loss = losses.BinaryCrossentropy(from_logits = False),
                     optimizer = 'adam',
                     metrics = ['accuracy']
)

# loss, accuracy = export_model.evaluate(raw_test_ds)
# print(f'Loss : {loss}, Accuracy : {accuracy}')

# # prediction on new data !!
# examples = [
#     "I'm happy!",
#     "Do you really think I'm happy?",
#     "What the fuck are you doing?",
#     "Someone died of the cancer.",
#     "The movie was terrible..."
# ]
# print(export_model.predict(examples))

while True:
    msg = input()
    ex = [msg]
    print(export_model.predict(ex))