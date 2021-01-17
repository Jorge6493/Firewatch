import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.models import model_from_json
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

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


train_path = 'fire/train'
valid_path = 'fire/valid'
test_path = 'fire/test'

training_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, rotation_range=30, height_shift_range=0.2, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(directory=train_path, target_size=(224,224), classes=['fire', 'no-fire'], class_mode='categorical', batch_size= 64)

validation_generator = validation_datagen.flow_from_directory(directory=valid_path, target_size=(224,224), classes=['fire', 'no-fire'], class_mode='categorical', batch_size= 16)

trainingSteps = train_generator.n/64
validationSteps = validation_generator.n/16



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

history = model.fit(train_generator, steps_per_epoch = trainingSteps, epochs = 50, validation_data = validation_generator, validation_steps = validationSteps, callbacks=[cp_callback])
#score = model.evaluate(history, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model.")

#load .json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model.")

loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
