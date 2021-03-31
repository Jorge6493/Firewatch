import cv2
import sys
import numpy as np
from tensorflow import keras

import queue, threading, time

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


VIDEO_URL = "http://192.168.1.131:8080/camera/livestream.m3u8"
# VIDEO_URL = './input.mp4'


cap = VideoCapture(0)
if (cap.cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)

while(True):
    time.sleep(.5)   # simulate time between events

    # read one frame
    # ret, frame = cap.read()
    # frame = cv2.flip(frame,0)

    frame = cap.read()

    # TODO: perform frame processing here
    img_array = keras.preprocessing.image.img_to_array(frame)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = np.array([img_array])

    # display frame
    cv2.imshow('frame',img_array[0])
    cv2.imshow('frametrue',frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cap.cap.release()
cv2.destroyAllWindows()