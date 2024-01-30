
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
from tensorflow.keras.optimizers import Adam
import os
from enum import Enum

INPUT_SHAPE = 128
SCALE = INPUT_SHAPE / 512.0
mmd_images_path = '../my_images/mmd'
hamed_images_path = '../my_images/hamed'

def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Modify as per your file types
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                names.append(filename)

    return images, names

class TYPES(Enum):
    TYPE1 = 'type1'
    TYPE2 = 'type2'
    TYPE3 = 'type3'
    TYPE4 = 'type4'
    TYPE5 = 'type5'
# folder_path = r'C:\Users\user\Desktop\my_images'
def scale_image(img, factor=1):
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))

outputs = []
mmd_images,mmd_names = load_images_from_folder(mmd_images_path)
hamed_images,hamed_names = load_images_from_folder(hamed_images_path)
for i in range(len(mmd_images)):
    mmd_images[i] = scale_image(mmd_images[i],SCALE)
    outputs.append(1)
for i in range(len(hamed_images)):
    hamed_images[i] = scale_image(hamed_images[i],SCALE)
    outputs.append(0)
output = np.array(outputs)
images = np.array(mmd_images + hamed_images)
mmd_images = []
hamed_images = []

def normalize_input_images(images):
    # Convert images to float32
    images = images.astype('float32')
    images = images / 127.0 - 1 
    return images
# ... (define and compile your model)

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
print("check7")
# Base model
x = base_model(input_layer)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)  # Increased number of neurons
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
coord_output = layers.Dense(1, activation='sigmoid', name='coord_output')(x)
final_model = models.Model(inputs=input_layer, outputs=coord_output )

final_model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
final_model.summary()


images = normalize_input_images(images)

train_images, test_images, train_output, test_output = train_test_split(images, output, test_size=0.2, random_state=42, shuffle=True)
final_model.fit(train_images, train_output, validation_data=(test_images, test_output), batch_size=16, epochs=10)

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
final_model.save('../model/model.h5')





