import cv2
import sys
import numpy as np
from tensorflow import keras


# VIDEO_URL = "http://192.168.1.131:8080/camera/livestream.m3u8"
VIDEO_URL = './input.mp4'


cap = cv2.VideoCapture(VIDEO_URL)
if (cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)

while(True):
    # read one frame
    ret, frame = cap.read()
    # frame = cv2.flip(frame,0)

    # TODO: perform frame processing here
    img_array = keras.preprocessing.image.img_to_array(frame)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = np.array([img_array])

    # display frame
    cv2.imshow('frame',img_array[0])
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()