import sys
import numpy as np
from PIL import ImageGrab
import time

import cv2
printscreen_pil =  ImageGrab.grab()
printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
.reshape((printscreen_pil.size[1],printscreen_pil.size[0],4))

x,y = 50,50
start_time = time.time()
training_start = time.time()
flipflop = True
while(True):
    tempScreen = np.copy(printscreen_numpy)
    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.rectangle(tempScreen,(x,y),(x+150,y+150),(0,255,0),2)
    cv2.imshow('window',tempScreen)
    if(time.time()-start_time > 0.3):
       if(x > 1250):
           flipflop = False
           y = y+75
       if(flipflop==True):
           x = x+75
       else:
           x = x -75
           if x <=250:
               y= y+75
               flipflop = True
       start_time = time.time()
    if (cv2.waitKey(25) & 0xFF == ord('q') ) or (time.time()-training_start>60):
        cv2.destroyAllWindows()
        break
