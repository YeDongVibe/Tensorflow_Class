import tensorflow as tf
import numpy as np

# input data 선언
A = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8]], dtype='float32')
A = A.reshape(1, 2, 4, 1) # batch, row, col, channel

# 모델 생성(input layer)
x = tf.keras.layers.Input(shape=A.shape[1:])
y = tf.keras.layers.UpSampling2D()(x) # size = (2, 3), [None, 4, 8, 1] -> A가 2차원 행렬이기에 row와 col에 각 2를 곱해줌.

u = tf.keras.layers.Reshape([8, 1])(x) # x를 받아서 8*1로 reshape. 
z = tf.keras.layers.UpSampling1D()(u) # size = 2, [None, 16, 1] -> row에 2를 곱해줌. 왜 2를 곱해주니? -> 기존 값을 2배로 불려주는것이 업샘플링의 성질. 즉 열을 2배로 늘려주는 것임. 

model = tf.keras.Model(inputs=x, outputs=[y, z])

output = model.predict(A)
print('A[0, :, :, 0] = ', A[0, :, :, 0])
print('output[0] = ', output[0][0, :, :, 0])
print('output[1] = ', output[1][0, :, 0])