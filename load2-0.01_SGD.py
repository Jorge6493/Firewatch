import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json, load_model
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
import itertools

test_path = 'fire/testNOtrain3'
batchSize = 5

test_batches = image.ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224), classes=['fire', 'no-fire'], batch_size=batchSize, shuffle=False)
testSteps = test_batches.n/batchSize

loaded_model = load_model("models/test_sgd/modeltrain1IV3-0.01")
print("Loaded model.")

class_names = ['fire', 'no-fire']

predictions = loaded_model.predict(x = test_batches, verbose = 0, steps = testSteps)

for i in predictions:

    
    score = tf.nn.softmax(i)


    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )   

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

cm_plot_labels = ['fire', 'no-fire']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')
