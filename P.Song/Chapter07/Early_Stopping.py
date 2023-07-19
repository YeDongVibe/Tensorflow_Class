import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


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
print(x_train.shape)


# Dense layer이용해 MLP(Multi Layer Perceptron) 구축 (fully connected layer)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28))) # input layer
model.add(tf.keras.layers.Dense(units=5, activation='sigmoid')) # hidden layer(노드 수 : 5개)
model.add(tf.keras.layers.Dense(units=10, activation='softmax')) # output layer(노드 수  : 10개)

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01) # learing rate : 학습률(증감 범위)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정 : 특정 조건에서 모델 조기 종료
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=1, mode='auto')
# min_delta : 최소 갱신값(val_loss 갱신 값이 min_delta범위까지 갈때) / patience : 위반 허용 횟수 / verbose : callback 메시지 출력 / mode : auto, min, max (각 해당될때 mode에서 값이 주어지면 모니터링 값 증가, 감소 할때 stop)

# 모델 학습
ret = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2, verbose=2, callbacks=[callback]) # batch_size : 연산 한번에 들어가는 데이터 크기 / validation_split : 검증 데이터로 사용할 학습 데이터의 비율 / epochs: 동작 세트 횟수

def scheduler(epoch, lr): # 처음에는 lr이 rough하게 설정하여 동작해도 무관하지만 동작을 진행할 수록 좀더 세밀한 범위로 가야지 값의 변동률이 급격해지지 않음
    if epoch % 2 == 0 and epoch:
        return 0.1*lr # 2에폭 기준으로 10%씩 lr을 감소함
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

ret = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.2, verbose=0, callbacks=[callback])

path = "C:/Ye_Dong/Tensorflow_Class/P.Song/tensorboard/"
if not os.path.isdir(path):
    os.mkdir(path)
logdir = path + "3101"

callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='epoch', histogram_freq=10, write_images=True)

ret = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2, verbose=2, callbacks=[callback])

