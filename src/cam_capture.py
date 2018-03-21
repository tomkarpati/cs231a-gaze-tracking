import sys
import os
sys.path.append(os.path.dirname(__file__)+"cnn")


import numpy as np
import cv2
import time
import thread
from PIL import ImageGrab
from threading import Thread
import pyautogui

import util
import pTfCNNPredictor as p

cnn_eye_size = [32, 32]
cnn_config = util.generate_cnn_config(cnn_eye_size)

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

predictor = p.pTfCNNPredictor(cnn_config,
                              tensorboard=True,
                              logging=True,
                              verbose=True)
#predictor.load()

cap = cv2.VideoCapture(0)
class facebox:
    x,y,w,h = 0,0,0,0
    def __init__(self, x=0,y=0,w=0,h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def toArray(self):
        return np.array([self.x,
                         self.y,
                         self.w,
                         self.h])

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

    frmcnt = 0

    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyebox_valid = [False, False]

        frmcnt += 1
        print "frame: {}".format(frmcnt)

        print "number of faces: {}".format(len(faces))
        for (x1,y1,w,h) in faces:
            roi_gray = gray[y1:y1+h, x1:x1+w]
            roi_color = img[y1:y1+h, x1:x1+w]
            eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
            print "number of eyes: {}".format(len(eyes))
            if(faces.shape == (1,4) and eyes != () and eyes.shape ==(2,4)):
                facebox_var = facebox(x1,y1,w,h)
                print x,y,w,h
                i = 0
                for (ex,ey,ew,eh) in eyes:
                    #print ex,ey,ew,eh
                    deltaX = facebox_var.x
                    deltaY = facebox_var.y
                    eyebox_var[i] = facebox(facebox_var.x+ex,facebox_var.y+ey,ew,eh)
                    eyebox_valid[i] = True
                    print eyebox_var[i].toArray()
                    i+=1
        verify_face(img,facebox_var,eyebox_var)

        #Process the eyes
        left = 0
        right = 1
        eyeFeaturesDim = [len(eyebox_var)]
        eyeFeaturesDim.extend(cnn_eye_size)
        eyeFeaturesDim.append(4)
        eyeFeatures = np.empty(eyeFeaturesDim)
        if (eyebox_valid[0] and eyebox_valid[1]):
            for i in range(len(eyebox_var)):
                e = eyebox_var[i]
                print "processing eye {}, {}".format(i,e.toArray())
                (eyeFeatures[i], drawn) = util.cnnProcessEyes(img[e.y:e.y+e.h,
                                                                  e.x:e.x+e.w,
                                                                  :],
                                                              cnn_eye_size)
                img[(i*cnn_eye_size[0]):(i*cnn_eye_size[0])+32,
                    0:cnn_eye_size[1]] = drawn
                
            # Swap the eyes as necessary
            if (eyebox_var[0].x > eyebox_var[1].x):
                left = 1
                right = 0
                print "SWAP......."


        if test == True:
            tempScreen = np.copy(printscreen_numpy)
            cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.rectangle(tempScreen,(x*2-75,y*2-75),(x*2+75,y*2+75),(0,255,0),-1)
            pyautogui.moveTo(x, y)
            cv2.imshow('window',tempScreen)
            if  time.time()-start_time > 1:
                if(x > 1200):
                    flipflopX = False
                if(flipflopX==True):
                    x = x + 25
                else:
                    x = x -25
                    if yMove == 0:
                        y = y + 15
                    elif yMove == -1:
                        y = y - 15
                    if x <=250:
                        if yMove == 0:
                            yMove = -1
                        else:
                            yMove = 0
                        flipflopX = True
                        start_time = time.time()

            if (eyebox_valid[0] and eyebox_valid[1]):
                predTarget = predictor.train_on_line(frmcnt,
                                                     util.reshapeToBatch(eyeFeatures[left,:,:,0:4]),
                                                     util.reshapeToBatch(eyeFeatures[right,:,:,0:4]),
                                                     util.reshapeToBatch(eyebox_var[left].toArray()),
                                                     util.reshapeToBatch(eyebox_var[right].toArray()),
                                                     util.reshapeToBatch(faces[0]),
                                                     util.reshapeToBatch(np.array([x, y]))) 
                print "target: {}, Pred Target: {}".format([x,y], predTarget)
               
                                                        
                                    
            #CNN_TRAIN(roi_color, eyebox_var,x,y)
            cv2.imshow('img',img)
        else:
            cv2.imshow('img',img)

            if (eyebox_valid[0] and eyebox_valid[1]):
                predTarget = predictor.train_on_line(frmcnt,
                                                     util.reshapeToBatch(eyeFeatures[left,:,:,0:4]),
                                                     util.reshapeToBatch(eyeFeatures[right,:,:,0:4]),
                                                     util.reshapeToBatch(eyebox_var[left].toArray()),
                                                     util.reshapeToBatch(eyebox_var[right].toArray()),
                                                     util.reshapeToBatch(faces[0]),
                                                     util.reshapeToBatch(np.array([x, y])))
                print "target: {}, Pred Target: {}".format([x,y], predTarget)
                deltaX = predTarget[0,0]
                deltaY = predTarget[0,1]
                #deltaX,deltaY = CNN_PREDICT(roi_color,eyebox_var)
                pyautogui.moveTo(deltaX, deltaY)

            
        k = cv2.waitKey(2) & 0xFF
        if (k == ord('q') ):
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            colorImage = time.strftime("%Y%m%d-%H%M%S")+'.png'
            cv2.imwrite(colorImage,roi_color)
            print colorImage
        elif (time.time()-training_start>60):
            cv2.destroyWindow('window')
            test = False


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

