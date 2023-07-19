import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# Batch Nomalisation: 노드들의 값들을 nomalisation 하는 것? => overfitting 문제를 해결

#1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#2: normalise images
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

#3: one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) #(60000, 10)
y_test = tf.keras.utils.to_categorical(y_test) #(10000, 10)

#4: build a model
init = tf.keras.initializers.RandomUniform(0.0, 1.0)
##act = 'relu'
act = tf.keras.layers.LeakyReLU(alpha=0.3)

n = 100
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=10, activation='softmax', kernel_initializer=init))
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01) 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#5: creates a summary file writer for the given log directory
path = "C:/Ye_Dong/Tensorflow_Class/P.Song/tensorboard/"

#7: train and evalute the model
ret = model.fit(x_train, y_train, epochs=101, batch_size=200, validation_split=0.2,
                verbose=2)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)