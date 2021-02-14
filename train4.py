import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, Dropout
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

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

# workon keras_tflow

train_path = 'fire/train'
valid_path = 'fire/valid'
# test_path = 'fire/test'

batchSize = 10

# train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size= batchSize, shuffle=True)
training_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest')
train_generator = training_datagen.flow_from_directory(train_path, target_size=(224,224), color_mode="grayscale",classes=['fire', 'no-fire'], class_mode='categorical', shuffle = True, batch_size = batchSize)

# valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=True)
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(valid_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], class_mode='categorical', shuffle = True, batch_size= batchSize)
# test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224), color_mode="grayscale", classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=False)

# assert train_batches.n == 780*2
# assert valid_batches.n == 220*2
# assert test_batches.n == 110*2
# assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# print('Train Batches = '+train_batches.n)
# print('Valid Batches = '+valid_batches.n)
# print('Test Batches = '+test_batches.n)

stepsPerEpoch = train_generator.n/batchSize
validationSteps = validation_generator.n/batchSize
# testSteps = test_batches.n/batchSize

imgs, labels = next(train_generator)

# This function will plot images in the form of a grid with 1 row and 10 columns where images are placed in each column.
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 10, figsize=(20,20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show(block = True)
    
# print(labels)
# plotImages(imgs)

# model = Sequential([
# 		Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,1)),
# 		MaxPool2D(pool_size=(2,2), strides=2),
# 		Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
# 		MaxPool2D(pool_size=(2,2), strides=2),
# 		Flatten(),
# 		Dense(units=2, activation='softmax'),
# ])

input_tensor = Input(shape=(224, 224, 1))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=20, verbose=2, steps_per_epoch=stepsPerEpoch, validation_steps=validationSteps)

# test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs)
# print(test_labels)

print('testbatches.classes')
print(test_batches.classes)
# print(train_batches.classes)

# print(valid_batches.classes)

predictions = model.predict(x=test_batches,verbose=0,steps = testSteps)

print('np')
print(np.round(predictions))

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


test_batches.class_indices

cm_plot_labels = ['fire', 'no-fire']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')
