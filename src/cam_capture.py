import numpy as np
import cv2
import time
import thread
from PIL import ImageGrab
from threading import Thread
import pyautogui


#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
class facebox:
    x,y,w,h = 0,0,0,0
    def __init__(self, x=0,y=0,w=0,h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

facebox_var = facebox()
eyebox_var = [facebox(),facebox()]
a_lock = thread.allocate_lock()

def desktop_drawing(facebox_var,eyebox_var):
    deltaX = 100
    deltaY = 150
    test = True
    printscreen_pil =  ImageGrab.grab()
    printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
    .reshape((printscreen_pil.size[1],printscreen_pil.size[0],4))
    keyboard = cv2.imread('mac_keyboard.jpg',0)
    #print keyboard.shape
    x,y = 50,50
    start_time = time.time()
    training_start = time.time()
    flipflopX = True
    yMove = 0
    while(True):
        if test == True:
            tempScreen = np.copy(printscreen_numpy)
            cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.rectangle(tempScreen,(x,y),(x+150,y+150),(0,255,0),2)
            cv2.imshow('window',tempScreen)
            if  time.time()-start_time > 1:
                if(x > 1250):
                    flipflopX = False
                if(flipflopX==True):
                    x = x + 75
                else:
                    x = x -75
                    if yMove == 0:
                        y = y + 75
                    elif yMove == -1:
                        y = y - 75
                    if x <=250:
                        if yMove == 0:
                            yMove = -1
                        else:
                            yMove = 0
                        flipflopX = True
                        start_time = time.time()
        else:
            cv2.imshow('img',img)
            #pyautogui.moveTo(deltaX, deltaY)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x1,y1,w,h) in faces:
            roi_gray = gray[y1:y1+h, x1:x1+w]
            roi_color = img[y1:y1+h, x1:x1+w]
            eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
            if(faces.shape == (1,4) and eyes != () and eyes.shape ==(2,4)):
                facebox_var = facebox(x1,y1,w,h)
                #print x,y,w,h
                i = 0
                for (ex,ey,ew,eh) in eyes:
                    #print ex,ey,ew,eh
                    deltaX = facebox_var.x
                    deltaY = facebox_var.y
                    eyebox_var[i] = facebox(facebox_var.x+ex,facebox_var.y+ey,ew,eh)
                    i+=1
        verify_face(img,facebox_var,eyebox_var)
        k = cv2.waitKey(25) & 0xFF
        if (k == ord('q') ):
            cv2.destroyAllWindows()
            break
        elif (time.time()-training_start>10):
            cv2.destroyWindow('window')
            test = False
        elif k == 115:
            colorImage = time.strftime("%Y%m%d-%H%M%S")+'.png'
            cv2.imwrite(colorImage,roi_color)
            print colorImage


def verify_face(img,facebox_var,eyebox_var):
    #if(eyebox_var[1].x+eyebox_var[1].w < eyebox_var[0].x  ):
    cv2.rectangle(img,(facebox_var.x,facebox_var.y),
                  (facebox_var.x+facebox_var.w,facebox_var.y+facebox_var.h),
                  (255,0,0),2)
    cv2.rectangle(img,(eyebox_var[0].x,eyebox_var[0].y),(eyebox_var[0].x+eyebox_var[0].w,eyebox_var[0].y+eyebox_var[0].h),(0,255,0),2)
    cv2.rectangle(img,(eyebox_var[1].x,eyebox_var[1].y),(eyebox_var[1].x+eyebox_var[1].w,eyebox_var[1].y+eyebox_var[1].h),(0,255,0),2)

desktop_drawing(facebox_var,eyebox_var)
#thread_test = Thread(target=desktop_drawing)
#thread_test.start()

'''while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
        if(faces.shape == (1,4) and eyes != () and eyes.shape ==(2,4)):
            facebox_var = facebox(x,y,w,h)
            #print x,y,w,h
            i = 0
            for (ex,ey,ew,eh) in eyes:
                #print ex,ey,ew,eh
                eyebox_var[i] = facebox(facebox_var.x+ex,facebox_var.y+ey,ew,eh)
                i+=1
    
    
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
    elif k == 115:
        colorImage = time.strftime("%Y%m%d-%H%M%S")+'.png'
        cv2.imwrite(colorImage,roi_color)
        print colorImage
    else:
        print "Captured Char : ",k

    verify_face(img,facebox_var,eyebox_var)
    with a_lock:
        print "Call CNN"
    #cv2.imshow('img',img)

#cv2.imwrite(colorImage+'2.png'),roi_gray)
cap.release()
cv2.destroyAllWindows()'''

