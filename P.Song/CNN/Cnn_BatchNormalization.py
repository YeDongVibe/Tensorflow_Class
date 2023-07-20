import tensorflow as tf
from keras import layers
from keras import models
import matplotlib.pyplot as plt

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32')
# x_test = x_test.reshape((10000, 28, 28, 1))
x_test  = x_test.astype('float32')
x_train /= 255.0 # [0.0, 1.0]
x_test  /= 255.0

#3: one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) #(60000, 10)
y_test = tf.keras.utils.to_categorical(y_test) #(10000, 10)

# one-hot encoding 
init = tf.keras.initializers.he_uniform()
act = tf.keras.layers.LeakyReLU(alpha=0.3)
#3: build a model
# 필터(커널) 수, 패치(커널) 사이즈, 활성화함수, 입력 데이터
model = tf.keras.Sequential()
model.add(input(x_train.shape[1:]))
model.add(layers.Conv2D(16, (3, 3), activation=act)) # 2차원 convolution적용., 흑백이라서 1임. color면 일반적으로 3
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D()) #pooling을 2*2로 지정


model.add(layers.Conv2D(32, (3, 3), activation=act))
model.add(layers.MaxPooling2D())
model.add(model.add(tf.keras.layers.Dropout(rate=0.2)))
# model.add(layers.Conv2D(64, (3, 3), activation=act))
model.add(layers.Flatten()) # 결과물을 1차원(벡터)로 변환

# 완전 연결 분류기
# model.add(layers.Dense(64, activation=act))
model.add(layers.Dense(10, activation='softmax'))

# 모델 설계 결과 요약
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(x_train, y_train, epochs=100, batch_size=400, validation_data=(x_test, y_test), verbose=0)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print('Train loss : ', train_loss)
print('Train acc : ', train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('test loss : ', test_loss)
print('test acc : ', test_acc)
plt.title("Without regularization by traing data in mnist")
plt.plot(ret.history['accuracy'], "b-", label = "train accuracy")
plt.plot(ret.history['val_accuracy'], "r-", label = "val accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc = "best")
plt.show()