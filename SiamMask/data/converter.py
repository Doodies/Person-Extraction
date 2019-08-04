import cv2
import os
print(os.getcwd())
path = os.getcwd()
vidcap = cv2.VideoCapture('smallv1.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        return hasFrames, image
        
    return hasFrames, None

sec = 0
frameRate = 0.9 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    print(count)
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success, img = getFrame(sec)
    