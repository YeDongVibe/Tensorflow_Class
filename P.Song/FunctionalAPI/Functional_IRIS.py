import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def load_Iris(shuffle=False) : 
    label = {'setosa' : 0,  'versicolor' : 1, 'virginica' : 2}
    data = np.loadtxt('')
    
    if shuffle : 
        np.random.shuffle(data)
    return data

def train_test_data_set(iris_data, test_rate=0.2) : 
    n = int(iris_data.shape[0]*(1-test_rate))
    x_train = iris_data[:n, :-1]
    y_train = iris_data[:n, :-1]
    
    x_test = iris_data[:n, :-1]
    y_test = iris_data[:n, :-1]
    return (x_train, y_train), (x_test, y_test)


iris_data = load_Iris(shuffle=True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(iris_data, test_rate=0.2)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = np.expand_dims(x_train, axis = 2)
x_test = np.expand_dims(x_test, axis=2)

def create_cnn1d(input_shape, num_class = 3) : 
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=10, kernel_size=4, activation='sigmoid')(inputs)
    x = layers.Dense(units=num_class, activation='softmax')(x)
    
    outputs = layers.Flatten()(x)
    model = tf.keras.Model(inputs = inputs, outputs= outputs)
    return model

model = create_cnn1d(input_shape=(4, 1))
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=100, verbose=0)

y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis=1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis=1), y_label)
print('Confusion_matrix(C)', C)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)