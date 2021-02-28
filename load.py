import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json, load_model
import numpy as np
import pathlib



#load .json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# loaded_model.load_weights("model.h5")
loaded_model = load_model("models/modeltrain1IV3")
print("Loaded model.")

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
img_path_url = "/home/jorge/keras_tf/Firewatch/testtest.jpg"
url = pathlib.Path(img_path_url).as_uri()

path = tf.keras.utils.get_file('testtest',origin=url)
img_height = 224
img_width = 224

img = keras.preprocessing.image.load_img(
    path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ['fire', 'no-fire']

print(
    "This image most likely belongs to {}"
    .format(class_names[np.argmax(score)])
)
