import tensorflow as tf
import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

A = np.array([[1, 2], 
              [3, 4]], dtype='float32')
A = A.reshape(-1, 2, 2, 1)

B = np.array([[5, 6], 
              [7, 8]], dtype='float32')
B = B.reshape(-1, 2, 2, 1)

C = np.array([1, 2, 3]).astype('float32')
C = C.reshape(-1, 3, 1, 1)

x = tf.keras.layers.Input(shape=A.shape[1:])
y = tf.keras.layers.Input(shape=B.shape[1:])
z = tf.keras.layers.Input(shape=C.shape[1:])

out3 = tf.keras.layers.Add()([x, y])
out4 = tf.keras.layers.Concatenate()([x, y]) # 행렬 두개 붙이기
out5 = tf.keras.layers.Dot(axes = -1)([x, y]) # 외적
out6 = tf.keras.layers.Dot(axes = -1)([x, z]) # 외적

out_list = [x, y, out3, out4, out5, out6]
model = tf.keras.Model(inputs = [x, y, z], outputs = out_list)

print('model.output_shape = ', model.output_shape)

output = model.predict([A, B, C])
for i in range(len(output)):
    print('Output[{}] = {}'.format(i, output[i]))