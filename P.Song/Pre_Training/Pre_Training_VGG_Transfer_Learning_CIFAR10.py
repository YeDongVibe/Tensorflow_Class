import tensorflow as tf
from keras.datasets import cifar10
from keras.layers   import Input, Dense, Flatten
from keras.applications.vgg16 import preprocess_input, VGG16
import numpy as np
import matplotlib.pyplot as plt

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') # (50000, 32, 32, 3)
x_test  = x_test.astype('float32')  # (10000, 32, 32, 3)

# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

# preprocessing, 'caffe', x_train, x_test: BGR
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#3: resize_layer
inputs = Input(shape=(32, 32, 3))
resize_layer = tf.keras.layers.Lambda( 
                    lambda img: tf.image.resize(img,(224, 224)))(inputs) 

#4:
vgg_model = VGG16(weights='imagenet', include_top= False, 
                     input_tensor= resize_layer) # input_tensor= inputs
vgg_model.trainable=False # 기존 학습을 손상시키지 안도록 False

                      
#4-1: output: classification
x = vgg_model.output
x = Flatten()(x)   # x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outs  = Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outs)
model.summary()

#5: train and evaluate the model
filepath = "C:/Ye_Dong/Tensorflow_Class/P.Song/PreTrainingFile/4404-model.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath, verbose=0, save_best_only=True)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=30, batch_size= 64,
                validation_split=0.2, verbose=0, callbacks = [cp_callback])
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#6: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].set_ylim(0, 1.1)
ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc='lower right')
fig.tight_layout()
plt.show()
