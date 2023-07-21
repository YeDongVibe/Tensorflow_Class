import tensorflow as tf
import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# 입력값 새성
A = np.array([1, 2, 3, 4, 5]).astype('float32')
B = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float32')
A = np.reshape(A, (1, -1, 1))
B = np.reshape(B, (1, -1, 1))

# 모델 생성
input_x = tf.keras.layers.Input(shape=A.shape[1:])
input_y = tf.keras.layers.Input(shape=B.shape[1:])


x = tf.keras.layers.MaxPool1D()(input_x) # kernel = 2
y = tf.keras.layers.MaxPool1D()(input_y)

pad = y.shape[1] - x.shape[1]
x = tf.keras.layers.ZeroPadding1D(padding = (0, pad))(x)

out2 = tf.keras.layers.Add()([x, y])
out3 = tf.keras.layers.Concatenate()([x, y]) # x와 y를 붙여주는 것
out4 = tf.keras.layers.Dot(axes = [1, 1])([x, y]) # 내적
out5 = tf.keras.layers.Dot(axes = -1)([x, y]) # 외적

out_list = [x, y, out2, out3, out4, out5]
model = tf.keras.Model(inputs = [input_x, input_y], outputs = out_list)

print('model.output_shape = ', model.output_shape)