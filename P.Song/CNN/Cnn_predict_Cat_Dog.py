from keras.models import load_model
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model2 = load_model("")
model2.summary()

img_path = "C:/Ye_Dong/Tensorflow_Class/P.Song/CNN/CnnData/Cat_and_Dog_re_Ge4.h5"



img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

print(img_tensor.shape)

plt.figure(figsize=(2,2))
plt.imshow(img_tensor[0])
plt.xticks([])
plt.yticks([])
plt.show()

layer_outputs = [layer.output for layer in model2.layers[:8]]

activation_model = models.Model(input=model2.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation_result = activations[0]
print(first_layer_activation_result.shape)

plt.figure(figsize=(2, 2))
plt.matshow(first_layer_activation_result[0, :, :, 21], cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(figsize=(2, 2))
plt.matshow(first_layer_activation_result[0, :, :, 2], cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.show()

layer_names = []
for layer in model2.later[:8] : 
    layer_names.append(layer.name)
print('layer 이름 : ',  layer_names)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations) : 
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :, col*images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            
            channel_image *=64
            channel_image /= 128
            channel_image = np.clip(channel_image, 0, 255).astype('unit8')
            display_grid[col*size : (col+1)*size, row*size:(row+1)*size] = channel_image
            
    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()