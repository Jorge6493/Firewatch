import cv2
import sys

url = "http://192.168.3.25:1935/live/myStream/playlist.m3u8"
 

capture = cv2.VideoCapture(url)
 
if not capture.isOpened():
    print(error)

 
# cv::namedWindow("Stream", CV_WINDOW_AUTOSIZE);
 
# cv::Mat frame;

#stream_enable
while (capture.isOpened):
    if (not frame = capture.read()): 
        #Error
    
    cv2.imshow("Stream", frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
