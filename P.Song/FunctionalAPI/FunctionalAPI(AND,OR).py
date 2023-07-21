import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]], dtype=np.float32)
y_and = np.array([[0], [0], [0], [1]], dtype=np.float32)
y_or = np.array([[0], [1], [1], [1]], dtype=np.float32)

x_and = layers.Input(shape = (2,))
out_and = layers.Dense(units = 1, activation = 'sigmoid', name = 'and')(x_and)

x_or = layers.Input(shape = (2,))
out_or = layers.Dense(units = 1, activation = 'sigmoid', name = 'or')(x_or) # output unit이 1

model = tf.keras.Model(inputs = [x_and, x_or], outputs = [out_and, out_or])
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


ret = model.fit(x = [X, X], y = [y_and, y_or], epochs=100, batch_size=4, verbose=0)
test = model.evaluate(x = [X, X], y = [y_and, y_or], verbose=0)

print('total loss = ', test[0])
print('AND : loss = {}, acc = {}'.format(test[1], test[3]))
print('OR : loss = {}, acc = {}'.format(test[2], test[4]))

plt.plot(ret.history['loss'], 'r--', label = 'loss')
plt.plot(ret.history['and_loss'], 'g--', label = 'and_loss')
plt.plot(ret.history['or_loss'], 'b--', label = 'or_loss')
plt.plot(ret.history['and_accuracy'], 'g-', label = 'and_accuracy')
plt.plot(ret.history['or_accuracy'], 'b-', label = 'or_accruacy')
plt.xlabel('epochs')
plt.ylabel('loss and accuracy')
plt.legend(loc='best')
plt.show()