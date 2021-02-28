import cv2
import sys
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json, load_model
import numpy as np
import pathlib
# import keyboard

# set async_mode to 'threading', 'eventlet', 'gevent' or 'gevent_uwsgi' to
# force a mode else, the best mode is selected automatically from what's
# installed
async_mode = None

import time
import socketio

from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask_cors import CORS

sio = socketio.Server(logger=True, async_mode=async_mode)

app = Flask(__name__)
CORS(app)

app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
app.config['SECRET_KEY'] = 'secret!'
thread = None



def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
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
        
        label = class_names[np.argmax(score)]
        
        sio.sleep(2)
        count += 1
        # sio.emit('my_response', {'data': 'Server generated event'})
        sio.emit('my_response', {'data': label + " " + str(count)})
        label = "none"
        


@app.route('/')
def index():
    global thread
    if thread is None:
        thread = sio.start_background_task(background_thread)
    return render_template('index.html')

@sio.event
def my_broadcast_event(sid, message):
    sio.emit('my_response', {'data': message['data']})

@sio.event
def disconnect_request(sid):
    sio.disconnect(sid)


@sio.event
def connect(sid, environ):
    sio.emit('my_response', {'data': 'Connected', 'count': 0}, room=sid)


@sio.event
def disconnect(sid):
    print('Client disconnected')

if __name__ == "__main__":

    # load .json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)

    # loaded_model.load_weights("model.h5")
    loaded_model = load_model("models/modeltrain1IV3")
    print("Loaded model.")

    VIDEO_URL = "http://192.168.1.131:8080/camera/livestream.m3u8"
    # VIDEO_URL = "http://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8"


    cap = cv2.VideoCapture(VIDEO_URL)
    if (cap.isOpened() == False):
        print('!!! Unable to open URL')
        sys.exit(-1)

    # retrieve FPS and calculate how long to wait between each frame to be display
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000/fps)
    print('FPS:', fps)
    # app.run(debug=True)

    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        app.run(threaded=True)
    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi
        eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi
        try:
            from geventwebsocket.handler import WebSocketHandler
            websocket = True
        except ImportError:
            websocket = False
        if websocket:
            pywsgi.WSGIServer(('', 5000), app,
                              handler_class = WebSocketHandler).serve_forever()
        else:
            pywsgi.WSGIServer(('', 5000), app).serve_forever()
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :5000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)


