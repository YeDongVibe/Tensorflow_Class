import tensorflow as tf
from keras.datasets import cifar10
from keras.layers   import Input, Dense, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions 
from keras.preprocessing import image # pip install pillow

import numpy as np
import matplotlib.pyplot as plt

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)


#2
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')      # (50000, 32, 32, 3)
x_test = x_test.astype('float32')        # (10000, 32, 32, 3)

# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# preprocessing, 'caffe', x_train, x_test: BGR
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#3: resize_layer
inputs = Input(shape = (32, 32, 3))
resize_layer = tf.keras.layers.Lambda( 
                    lambda img: tf.image.resize(img,(224, 224)))(inputs)
res_model = ResNet50(weights = 'imagenet', include_top = False,
                 input_tensor = inputs)#resize_layer)       # inputs
res_model.trainable=False

#4: create top for cifar10 classification
x = res_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
outs  = Dense(10, activation = 'softmax')(x)
model = tf.keras.Model(inputs = inputs, outputs=outs)
model.summary()

#5: train and evaluate the model
filepath = "C:/Ye_Dong/Tensorflow_Class/P.Song/PreTrainingFile/4603-model.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                     filepath, verbose = 0, save_best_only = True)
                 
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
ret = model.fit(x_train, y_train, epochs = 30, batch_size = 32, 
                validation_split = 0.2, verbose = 2, callbacks = [cp_callback])
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)

#6: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].set_ylim(0, 1.1)
ax[1].plot(ret.history['accuracy'], "b-", label = "train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label = "val_accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc = 'lower right')
fig.tight_layout()
plt.show()
