
import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras import models
from keras.applications import MobileNetV2
from keras import layers
from labeling import TYPES, load_images_from_folder
from manipulate_img import scale_image

def normalize_input_images(images):
    # Convert images to float32
    images = images.astype('float32')
    images = images / 127.0 - 1 
    return images

OUTPUT_ELEMENT_SIZE = 10
INPUT_SHAPE = 128
SCALE = INPUT_SHAPE / 512.0
mmd_images_path = './../my_images/mmd'
hamed_images_path = './../my_images/hamed'


outputs = []
mmd_images,mmd_names = load_images_from_folder(mmd_images_path)
hamed_images,hamed_names = load_images_from_folder(hamed_images_path)

print(len(mmd_images),"mmd images")
print(len(hamed_images),"hamed images")


    
normal_images = []
augmented_images = []

normal_outputs = []
augmented_outputs = []

for i in range(len(hamed_images)):
    # stateless_random_contrast = tf.image.stateless_random_contrast(images[i], lower=0.1, upper=0.9, seed=(i,0))
    # stateless_random_contrast = scale_image(np.array(stateless_random_contrast), SCALE)
    # stateless_random_brightness = tf.image.stateless_random_brightness(images[i], max_delta=0.95, seed=(i,0))
    # stateless_random_brightness = scale_image(np.array(stateless_random_brightness),SCALE)
    
    
    hamed_images[i] = scale_image(hamed_images[i], SCALE)
    splitted_name = hamed_names[i].split('_')
    splitted_name.pop()  #drop the last part
    output_element = OUTPUT_ELEMENT_SIZE * [0]
    normal_images.append(hamed_images[i])

    for type_coordinates in splitted_name: #type_coordinates format => 'type1-305-289'
        [point_type, x, y] = type_coordinates.split('-')

        for index, currentType in enumerate(TYPES):
            if currentType.value == point_type:
                output_element[index*2] = float(x)
                output_element[index*2+1] = float(y)
                # output_element[10] = 0
                # augmented_images.append(stateless_random_contrast)
                # augmented_images.append(stateless_random_brightness)
    print("hamed", output_element)
    normal_outputs.append(output_element)
                
                # augmented_outputs.append(output_element)
                # augmented_outputs.append(output_element)
print("df")           
for i in range(len(mmd_images)):   
    mmd_images[i] = scale_image(mmd_images[i], SCALE)
    splitted_name = mmd_names[i].split('_')
    splitted_name.pop()  #drop the last part
    output_element = OUTPUT_ELEMENT_SIZE * [0]
    normal_images.append(mmd_images[i])   

    for type_coordinates in splitted_name: #type_coordinates format => 'type1-305-289'
        [point_type, x, y] = type_coordinates.split('-')

        for index, currentType in enumerate(TYPES):
            if currentType.value == point_type:
                output_element[index*2] = float(x)
                output_element[index*2+1] = float(y)
                # output_element[10] = 1
    print("mohammad", output_element)
           
    normal_outputs.append(output_element)
                
               

output = np.array(normal_outputs+augmented_outputs) / 512.0
images = np.array(normal_images+augmented_images)

print("images", len(images))
print("outputs", len(output))
normal_images = []
augmented_images = []
normal_outputs = []
augmented_outputs = []

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor= 'loss',  # You can use 'loss' if you want to monitor the training loss
                               patience=3,          # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restores model weights from the epoch with the best value of the monitored quantity
                               verbose=1) 

base_model = MobileNetV2(input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3), include_top=False, weights='imagenet')
# Fine-tuning: Unfreeze some of the last layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Freezing all layers except the last 20
    layer.trainable = False
# Input layer
input_layer = layers.Input(shape=(INPUT_SHAPE, INPUT_SHAPE, 3))

# Base model
x = base_model(input_layer)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)  # Increased number of neurons
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
coord_output = layers.Dense(10, activation='sigmoid', name='coord_output')(x)
final_model = models.Model(inputs=input_layer, outputs=coord_output )

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
final_model.summary()


images = normalize_input_images(images)

train_images, test_images, train_output, test_output = train_test_split(images, output, test_size=0.1, random_state=42, shuffle=True)
final_model.fit(train_images, train_output, validation_data=(test_images, test_output), batch_size=8, epochs=50)

plt.plot(final_model.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(final_model.history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# Save the model
final_model.save('models/model.h5')





