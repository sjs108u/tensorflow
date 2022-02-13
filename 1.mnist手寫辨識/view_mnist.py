import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path = 'C:\\Users\\User\\Desktop\\test\\mnist.npz')

print(len(x_test))

for idx in range(10):
    train_image = x_test[idx]

    for i in train_image:
        for j in i:
            if j > 0:
                print(1, end = '')
            else:
                print(0, end = '')
        print('')


print(y_test[1])

plt.figure()
plt.imshow(x_test[1])
plt.colorbar()
plt.grid(False)
plt.show()