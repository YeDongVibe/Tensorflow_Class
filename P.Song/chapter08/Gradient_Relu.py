import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # mnist는 손글씨데이터(0~9까지 이루어져있음)

# normalize images
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# build a model
# init = tf.keras.initializers.he_normal()
# act = tf.keras.activations.relu

# init = tf.keras.initializers.he_normal()
# act = tf.keras.layers.LeakyReLU(alpha=0.3) # alpha : 음수쪽의 기울기를 지정

init = tf.keras.initializers.he_uniform()
act = tf.keras.layers.LeakyReLU(alpha=0.3)

n = 100
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28))) # input layer
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=10, activation='softmax', kernel_initializer=init)) # output layer(노드 수  : 10개)
model.summary()
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# path = "C:/Ye_Dong/Tensorflow_Class/tensorboard/"
# if not os.path.isdir(path):
#     os.mkdir(path)
# logdir = path + "3203"

# file_writer = tf.summary.create_file_writer(logdir + "/gradient")
# file_writer.set_as_default()

# callback1 = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=10)

ret = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2, verbose=2)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# class GradientCallback(tf.keras.callbacks.Callback):
#     def __init__(self, freq=10):
#         self.freq = freq
        
#     def on_epoch_end(self, epoch, logs):
#         if epoch%self.freq !=0:
#             return
#         with tf.GradientTape() as tape:
#             y_pred = model(x_train)
#             loss = tf.keras.losses.binary_crossentropy(y_train, y_pred)
#         grads = tape.gradient(loss, model.trainable_weights)
#         for n in range(1, len(model.layers)):
#             i2 = (n-1)*2
#             i1 = i2 + 1
            
#             bias_avg = tf.reduce_mean(tf.abs(grads[i1]))
#             weight_avg = tf.reduce_mean(tf.abs(grads[i2]))
            
#             tf.summary.scalar("layer_%d/avg/bias"%n, )
            
            
        