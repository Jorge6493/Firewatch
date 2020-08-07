import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


train_path = 'fire/train'
valid_path = 'fire/valid'
test_path = 'fire/test'

batchSize = 10

train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size= batchSize)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size=batchSize)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=False)

assert train_batches.n == 780*2
assert valid_batches.n == 220*2
assert test_batches.n == 110*2
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# print('Train Batches = '+train_batches.n)
# print('Valid Batches = '+valid_batches.n)
# print('Test Batches = '+test_batches.n)

stepsPerEpoch = train_batches.n/batchSize
validationSteps = valid_batches.n/batchSize

imgs, labels = next(train_batches)

# This function will plot images in the form of a grid with 1 row and 10 columns where images are placed in each column.
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 10, figsize=(20,20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show()
    
# plotImages(imgs)
# print(labels)

model = Sequential([
		Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,1)),
		MaxPool2D(pool_size=(2,2), strides=2),
		Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
		MaxPool2D(pool_size=(2,2), strides=2),
		Flatten(),
		Dense(units=2, activation='softmax'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2, steps_per_epoch=stepsPerEpoch, validation_steps=validationSteps)

# predict steps = stepsPerEpoch