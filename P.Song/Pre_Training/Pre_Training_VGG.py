import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

X = x_train.copy()
Y1 = preprocess_input(X)
del X

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(x_train[0, :, :, :].astype(np.uint8))
ax[0].set_title('x_train[0] : RGB')
ax[0].axis("off")

mean = np.array([103.939, 166.779, 123.68], dtype=np.float32)
Y1 +=mean

ax[1].
ax[1].
ax[1].
fig.tight_layout()
plt.show()