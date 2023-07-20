import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# Add channel dimension for convolutional layers
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

# One-hot encoding of the target labels for training data
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# L2 Regularization parameter
l2_reg = 0.01

# Build the model with L2 regularization
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
opt = 'rmsprop'
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
ret = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print('Train loss : ', train_loss)
print('Train acc : ', train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('test loss : ', test_loss)
print('test acc : ', test_acc)

# Plot the training and validation accuracy
plt.title("Accuracy with L2 Regularization")
plt.plot(ret.history['accuracy'], "b-", label="train accuracy")
plt.plot(ret.history['val_accuracy'], "r-", label="val accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc="best")
plt.show()
