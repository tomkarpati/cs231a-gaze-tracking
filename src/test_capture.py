import sys
import os
sys.path.append(os.path.dirname(__file__)+"cnn")

import numpy as np
import cv2
import time

import util
import pTfCNNPredictor as p

cnn_eye_size = [32, 32]

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


space = {}

image_space = {}
image_space['image_x'] = 32
image_space['image_y'] = 32
image_space['image_c'] = 4
space['image'] = image_space

space['num_conv_layers'] = 3

conv_layer_0_space = {}
conv_layer_0_space['filter'] = [ 3, 32 ]
conv_layer_0_space['stride'] = 1
conv_layer_0_space['pooling'] = [1, 1]
space['conv_layer_0'] = conv_layer_0_space

conv_layer_1_space = {}
conv_layer_1_space['filter'] = [ 5, 64 ]
conv_layer_1_space['stride'] = 1
conv_layer_1_space['pooling'] = [2, 2]
space['conv_layer_1'] = conv_layer_1_space

conv_layer_2_space = {}
conv_layer_2_space['filter'] = [ 5, 64 ]
conv_layer_2_space['stride'] = 1
conv_layer_2_space['pooling'] = [2, 2]
space['conv_layer_2'] = conv_layer_2_space

space['num_fc_layers'] = 3
space['fc_layer_0'] = 1024
space['fc_layer_1'] = 1024
space['fc_layer_2'] = 1024
space['readout_vec'] = 2


predictor = p.pTfCNNPredictor(space,
                              tensorboard=True,
                              logging=True,
                              verbose=True)
predictor.load()

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

target = [0, 0]

counter  = 0
train = False
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    imgDim = np.shape(img)

    points = [[int(0.1*imgDim[1]), int(0.1*imgDim[0])],
              [int(0.5*imgDim[1]), int(0.1*imgDim[0])],
              [int(0.9*imgDim[1]), int(0.1*imgDim[0])],
              [int(0.1*imgDim[1]), int(0.5*imgDim[0])],
              [int(0.5*imgDim[1]), int(0.5*imgDim[0])],
              [int(0.9*imgDim[1]), int(0.5*imgDim[0])],
              [int(0.1*imgDim[1]), int(0.9*imgDim[0])],
              [int(0.5*imgDim[1]), int(0.9*imgDim[0])],
              [int(0.9*imgDim[1]), int(0.9*imgDim[0])]]

    if ((counter % 10) == 0):
        print len(points)
        
        choice = np.random.choice(len(points),1)
        print choice[0]
        target = points[choice[0]]
        
    #target[0] += 25
    #if (target[0] >= imgDim[1]):
    #    target[0]  = 0
    #    target[1]  = np.random.randint(0,imgDim[0])
    #    if (target[1] >= imgDim[0]):
    #        target[1] = 0

    bb0 = (target[0], target[1])
    bb1 = (target[0]+10, target[1]+10)
    
    print "sizes"
    print imgDim
    print bb0
    print bb1
    cv2.rectangle(img, bb0, bb1, (255, 0, 0), 5)
            

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
                
        # run Hough transform for pupil detector
        eyeFeaturesDim = [len(eyes)]
        eyeFeaturesDim.extend(cnn_eye_size)
        eyeFeaturesDim.append(4)
        eyeFeatures = np.empty(eyeFeaturesDim)
        print "Found {} eyes".format(len(eyes))
        for i in range(len(eyes)):
            e = eyes[i]
            print "Eye {}: {}".format(i,e)
            (eyeFeatures[i], drawn) = util.cnnProcessEyes(roi_color[e[1]:e[1]+e[3],
                                                                     e[0]:e[0]+e[2],
                                                                     :],
                                                           cnn_eye_size)
            img[(i*cnn_eye_size[0]):(i*cnn_eye_size[0])+32,
                0:cnn_eye_size[1]] = drawn

        if (len(eyes) > 1):
            left = 0
            right = 1
            if (eyes[0][0] > eyes[1][0]):
                left = 1
                right = 0
                print "SWAP***********************"
                
            eyeLBB = eyes[left]
            eyeLBB[0] += x
            eyeLBB[1] += y
            eyeRBB = eyes[right]
            eyeRBB[0] += x
            eyeRBB[1] += y
            if (train):
                predTarget = predictor.train_on_line(counter,
                                                     util.reshapeToBatch(eyeFeatures[left,:,:,0:4]),
                                                     util.reshapeToBatch(eyeFeatures[right,:,:,0:4]),
                                                     util.reshapeToBatch(np.array(eyeLBB)),
                                                     util.reshapeToBatch(np.array(eyeRBB)),
                                                     util.reshapeToBatch(np.array([x,y,w,h])),
                                                     util.reshapeToBatch(np.array(target)))
            else:
                predTarget = predictor.predict(counter,
                                               util.reshapeToBatch(eyeFeatures[left,:,:,0:4]),
                                               util.reshapeToBatch(eyeFeatures[right,:,:,0:4]),
                                               util.reshapeToBatch(np.array(eyeLBB)),
                                               util.reshapeToBatch(np.array(eyeRBB)),
                                               util.reshapeToBatch(np.array([x,y,w,h])),
                                               util.reshapeToBatch(np.array(target)))
                
            print "counter: target: {}, predTarget: {}".format(target, predTarget)

            predTarget[0,0] = max(0, predTarget[0,0])
            predTarget[0,0] = min(imgDim[1], predTarget[0,0])
            predTarget[0,1] = max(0, predTarget[0,1])
            predTarget[0,1] = min(imgDim[0], predTarget[0,1])
            
            cv2.circle(img, (int(predTarget[0,0]), int(predTarget[0,1])), 10, (0, 0, 255), 5)


    k = cv2.waitKey(1)
    if k == ord('s'):
        print "CAlling method to store the predictor"
        predictor.save()
    elif k == ord('t'):
        print "Switch to testing"
        train = not train
    elif k == 27:
        break
    elif k == 115:
        colorImage = time.strftime("%Y%m%d-%H%M%S")+'.png'
        cv2.imwrite(colorImage,roi_color)
        print colorImage
    else:
        print "Captured Char : ",k

    verify_face(img,facebox_var,eyebox_var)

    cv2.imshow('img',img)

    counter += 1

#cv2.imwrite(colorImage+'2.png'),roi_gray)
cap.release()
cv2.destroyAllWindows()

