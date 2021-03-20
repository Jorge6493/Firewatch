import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json, load_model
import numpy as np
import pathlib
import cv2


# loaded_model = load_model("models/modeltrain1IV3")
# print("Loaded model.")

# VIDEO_URL = './input.mp4'
# cap = cv2.VideoCapture(VIDEO_URL)
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)
print('wait_ms:', wait_ms)

class_names = ['fire', 'no-fire']
img_height = 224
img_width = 224
# frameskip = False
frame = True
count = 0
while(cap.isOpened()):

    # if(not frameskip):
        # read one frame
        ret, frame = cap.read()
        if ret:
            # cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            # count += fps # i.e. at 30 fps, this advances one second
            # cap.set(1, count)

            # follow link to do more
            # https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    
            frame = cv2.resize(frame, (img_height, img_width))

            # TODO: perform frame processing here

            img_array = keras.preprocessing.image.img_to_array(frame)
            # Create a batch
            img_array = np.array([img_array])


            # display frame
            cv2.imshow('interpreted frame',img_array[0])
            cv2.imshow('real frame',frame)

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

            # predictions = loaded_model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])

            # print(
            #     "This image most likely belongs to {}"
            #     .format(class_names[np.argmax(score)])
            # )
        else:
            break
        

    #     frameskip = True
    # frameskip = False


cap.release()
cv2.destroyAllWindows()

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
# img_path_url = "/home/jorge/keras_tf/Firewatch/testtest.jpg"
# url = pathlib.Path(img_path_url).as_uri()

# path = tf.keras.utils.get_file('testtest',origin=url)
# img_height = 224
# img_width = 224

# img = keras.preprocessing.image.load_img(
#     path, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# # Create a batch
# img_array = np.array([img_array])
# while(True):
#     cv2.imshow('frame',img_array[0])
#     if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
#         break


