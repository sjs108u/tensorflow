# reference : https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh-tw

from audioop import findmax
from pickletools import optimize
import tensorflow as tf

# find the index of number which is the largest one in array
# ex:
# array = [9, 1, 3, 4], the biggest number is 9,
# so findMaxIndex(array) returns index = 0
def findMaxIndex(array):
    idx = 0
    max = array[0]
    # len(array[0]) contains 0 ~ 9, so equals 10
    for i in range(len(array)):
        if array[i] > max:
            max = array[i]
            idx = i
        # print(array[i])
    return idx
        


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path = 'C:\\Users\\User\\Desktop\\test\\mnist.npz')

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax'),
])

# 編譯模型

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
"""

# 訓練模型
model.fit(x_train, y_train, epochs = 5)


loss, accuracy = model.evaluate(x_test, y_test)
print('Test:')
print(f'Loss:{loss}, Accuracy:{accuracy}')

for idx in range(10):
    prediction = model.predict(x_test[idx:idx+1])
    softmax = tf.nn.softmax(prediction)

    # print('prediction:')
    # print(prediction)

    # print('softmax:')
    # print(softmax)

    train_image = x_test[idx]
    for i in train_image:
        for j in i:
            if j > 0:
                print(1, end = '')
            else:
                print(0, end = '')
        print('')

    myans = findMaxIndex(prediction[0])
    print(f'my answer:{myans}')
    print(f'answer:{str(y_test[idx])}')