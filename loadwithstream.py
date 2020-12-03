import cv2
import sys
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import pathlib
# import keyboard

# from flask import Flask, render_template, url_for, request, redirect

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')
#     return "Hello World"


#load .json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model.")

VIDEO_URL = "http://192.168.1.131:8080/camera/livestream.m3u8"

cap = cv2.VideoCapture(VIDEO_URL)
if (cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)

# if __name__ == "__main__":
#     app.run(debug=True)

while(True):
        # read one frame
        ret, frame = cap.read()

        # TODO: perform frame processing here
        img_height = 224
        img_width = 224
        frame = cv2.resize(frame, (img_height, img_width))
        # frame = cv2.flip(frame,0) 
        # img = keras.preprocessing.image.load_img(
        # frame, target_size=(img_height, img_width)
        # )
        img_array = keras.preprocessing.image.img_to_array(frame)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['fire', 'no-fire']

        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        # display frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

# cv2.destroyAllWindows()