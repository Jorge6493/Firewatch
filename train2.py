import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#workon dl4cv

train_path = 'fire/train'
valid_path = 'fire/valid'
test_path = 'fire/test'

training_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, rotation_range=30, height_shift_range=0.2, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(directory=train_path, target_size=(224,224), classes=['fire', 'no-fire'], class_mode='categorical', batch_size= 64)

validation_generator = validation_datagen.flow_from_directory(directory=valid_path, target_size=(224,224), classes=['fire', 'no-fire'], class_mode='categorical', batch_size= 16)


from tensorflow.keras.optimizers import Adam
model = Sequential([
Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), activation='relu', input_shape=(224,224,3)), 
MaxPool2D(pool_size=(3,3), strides=(2,2)), 
Conv2D(filters=256, kernel_size=(5, 5), activation='relu'), 
MaxPool2D(pool_size=(3,3), strides=(2,2)), 
Conv2D(filters=384, kernel_size=(5, 5), activation='relu'), 
MaxPool2D(pool_size=(3,3), strides=(2,2)), 
Flatten(), 
Dropout(0.2),
Dense(2048, activation='relu'), 
Dropout(0.25),
Dense(1024, activation='relu'), 
Dropout(0.2),
Dense(2, activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch = 15, epochs = 50, validation_data = validation_generator, validation_steps = 15)

