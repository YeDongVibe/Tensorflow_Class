import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = VGG16(weights='imagenet', include_top=True)
model.summary()

img_path = 'C:/Ye_Dong/Tensorflow_Class/P.Song/DATA/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
x = preprocess_input(x) # mode='caffe'
output = model.predict(x)

print('Predicted:', decode_predictions(output, top=5)[0]) # 상위 5개 뽑아서 나옴

plt.imshow(img)
plt.axis("off")
plt.show()