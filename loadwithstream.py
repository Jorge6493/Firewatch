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

import queue, threading

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

sio = socketio.Server(logger=True, async_mode=async_mode)

app = Flask(__name__)
CORS(app)

app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
app.config['SECRET_KEY'] = 'secret!'
thread = None



def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    count2 = 0
    img_height = 224
    img_width = 224
    class_names = ['fire', 'no-fire']


    while True:
        # read one frame
        frame = cap.read()
        if True:
            # cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            # count2 += wait_ms*2 # i.e. at 30 fps, this advances one second
            # cap.set(1, count2)
    
            frame = cv2.resize(frame, (img_height, img_width))

            # TODO: perform frame processing here

            img_array = keras.preprocessing.image.img_to_array(frame)
            # Create a batch
            img_array = np.array([img_array])


            # display frame
            cv2.imshow('frame interpretado',img_array[0])
            cv2.imshow('frame real',frame)

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

            predictions = loaded_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            label = class_names[np.argmax(score)]
            
            sio.sleep(2)
            count += 1
            # sio.emit('my_response', {'data': 'Server generated event'})
            sio.emit('my_response', {'data': label + " " + str(count)})
            print(label + " " + str(count))
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

    # VIDEO_URL = "http://192.168.1.131:8080/camera/livestream.m3u8"
    # VIDEO_URL = "http://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8"
    VIDEO_URL = "https://media.publit.io/file/h_720/input.mp4"
    # VIDEO_URL = './input.mp4'


    cap = VideoCapture(VIDEO_URL)
    if (cap.cap.isOpened() == False):
        print('!!! Unable to open URL')
        sys.exit(-1)

    # retrieve FPS and calculate how long to wait between each frame to be display
    fps = cap.cap.get(cv2.CAP_PROP_FPS)
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


