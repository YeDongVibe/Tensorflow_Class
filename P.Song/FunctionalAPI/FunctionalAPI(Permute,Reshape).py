import tensorflow as tf
import numpy as np

A = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8]], dtype='float32')
A = A.reshape(1, 2, 4, 1) # batch, row, col, channel


# 모델 생성(input layer)
x = tf.keras.layers.Input(shape=A.shape[1:])
y = tf.keras.layers.Reshape([4, 2, 1])(x) # 데이터 형태 reshape = [1, 4, 2, 1]
z = tf.keras.layers.Permute([2, 1, 3])(x)

model = tf.keras.Model(inputs=x, outputs=[y, z])
model.summary()

output = model.predict(A)
print('A[0, :, :, 0] = ', A[0, :, :, 0])
print('output[0] = ', output[0][0, :, :, 0])
print('output[1] = ', output[1][0, :, :, 0])