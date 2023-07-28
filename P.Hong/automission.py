import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.cifar10.load_data()
print(train_X.shape, train_Y.shape)
print(train_X.dtype)
train_X = train_X / 255.0
test_X = test_X / 255.0

plt.imshow(train_X[0])
plt.colorbar()
plt.show()

print(train_Y[0])