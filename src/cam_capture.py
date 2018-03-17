import numpy as np
import cv2
import time

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

def verify_face(img,facebox_var,eyebox_var):
    #if(eyebox_var[1].x+eyebox_var[1].w < eyebox_var[0].x  ):
    cv2.rectangle(img,(facebox_var.x,facebox_var.y),
                  (facebox_var.x+facebox_var.w,facebox_var.y+facebox_var.h),
                  (255,0,0),2)
    cv2.rectangle(img,(eyebox_var[0].x,eyebox_var[0].y),(eyebox_var[0].x+eyebox_var[0].w,eyebox_var[0].y+eyebox_var[0].h),(0,255,0),2)
    cv2.rectangle(img,(eyebox_var[1].x,eyebox_var[1].y),(eyebox_var[1].x+eyebox_var[1].w,eyebox_var[1].y+eyebox_var[1].h),(0,255,0),2)

while 1:
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

    cv2.imshow('img',img)

#cv2.imwrite(colorImage+'2.png'),roi_gray)
cap.release()
cv2.destroyAllWindows()

