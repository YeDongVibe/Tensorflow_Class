import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions 
from keras.preprocessing import image

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
model = ResNet50(weights='imagenet', include_top=True) # weights= W
model.summary()

#3: predict an image
#3-1
img_path = 'C:/Ye_Dong/Tensorflow_Class/P.Song/DATA/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) # (1, 224, 224, 3)
x = preprocess_input(x)
output = model.predict(x)

#3-2: (class_name, class_description, score)
top5 = decode_predictions(output, top=5)[0]
print('Top-5 predicted:', top5)
#direct Top-1, and Top-5, ref[4502]

#4: display image and labels  
plt.imshow(img)
plt.title(top5[0][1])
plt.axis("off")
plt.show()
