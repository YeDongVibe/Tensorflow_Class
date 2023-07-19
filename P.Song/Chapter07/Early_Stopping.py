import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=1, mode='auto')

ret = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2, verbose=2, callbacks=[callback])

def scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch:
        return 0.1*lr
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

ret = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.2, verbose=0, callbacks=[callback])