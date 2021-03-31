import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
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

learning_rate = 0.0001

# workon keras_tflow

train_path = 'fire/train'
valid_path = 'fire/valid'
test_path = 'fire/test'

batchSize = 10

train_batches = image.ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224), classes=['fire', 'no-fire'], batch_size= batchSize, shuffle=True)
valid_batches = image.ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224), classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=True)
test_batches = image.ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224), classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=False)



# assert train_batches.n == 780*2
# assert valid_batches.n == 220*2
# assert test_batches.n == 110*2
# assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# print('Train Batches = '+train_batches.n)
# print('Valid Batches = '+valid_batches.n)
# print('Test Batches = '+test_batches.n)

stepsPerEpoch = train_batches.n/batchSize
validationSteps = valid_batches.n/batchSize
testSteps = test_batches.n/batchSize

imgs, labels = next(train_batches)

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

#original
# model = Sequential([
# 		Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,1)),
# 		MaxPool2D(pool_size=(2,2), strides=2),
# 		Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
# 		MaxPool2D(pool_size=(2,2), strides=2),
# 		Flatten(),
# 		Dense(units=2, activation='softmax'),
# ])

#incelption
model = Sequential([
		Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,1)),
		MaxPool2D(pool_size=(2,2), strides=2),
		Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
		MaxPool2D(pool_size=(2,2), strides=2),
		Flatten(),
		Dense(units=1, activation='sigmoid'),
])

model.summary()

model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=20, verbose=2, steps_per_epoch=stepsPerEpoch, validation_steps=validationSteps)

# model_json = model.to_json()
# with open("modeltrain1.json", "w") as json_file:
# 	json_file.write(model_json)

# model.save_weights("modeltrain1.h5")
model.save("models/test_binaryCross/modeltrain1IV3-binarCross")
print("Saved model.")



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

