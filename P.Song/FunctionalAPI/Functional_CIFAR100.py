import tensorflow as tf
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train) #(60000, 10)
y_test = tf.keras.utils.to_categorical(y_test)

def normalize_image(image):
    mean = np.mean(image, axis=(0, 1, 2))
    std = np.std(image, axis=(0, 1, 2))
    image = (image-mean)/std
    return image
x_train = normalize_image(x_train)
x_test = normalize_image(x_test)

def create_cnn2d(input_shape, num_class = 10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.25)(x)
    
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.5)(x)
    
    x = layers.Flatten()(x)
    
    outputs = tf.keras.layers.Dense(units=num_class, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = create_cnn2d(input_shape=x_train.shape[1:])

opt = tf.keras.optimizers.RMSprop(learning_rate=0.001) 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(x_train, y_train, epochs=200, batch_size=400, validation_data=(x_test, y_test), verbose=0)

y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis=1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis=1), y_label)
print('Confusion_matrix(C)', C)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'], 'g-')
ax[0].set_title('train loss')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'], 'b-', label='train accruacy')
ax[1].plot(ret.history['val_accuracy'], 'r', label='test accruacy')
ax[1].set_title('accuracy')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.show()