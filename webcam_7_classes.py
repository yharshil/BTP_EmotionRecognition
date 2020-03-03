# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# =============================================================================
# This module is for testing Saved Images
# Will extend to frames from Live Feed
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Input Image Preprocessing Image Shape = (48,48,1) save it in variable X

# Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils

classes = 7

model = Sequential()
# =============================================================================
# Section 1
# =============================================================================
model.add(Convolution2D(128,(4,4),input_shape=(48,48,1),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 2
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 3
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))

# =============================================================================
# Section 4
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 5
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))

# =============================================================================
# Section 6
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 7
# =============================================================================
model.add(Convolution2D(128,(2,2),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))


model.add(Flatten())

# =============================================================================
# Section 8
# =============================================================================
model.add(Dense(1024))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 9
# =============================================================================
model.add(Dense(1024))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 10
# =============================================================================
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

fname = '/Users/harshilyadav/Desktop/weights.best.acc.221_model.hdf5'
model.load_weights(fname)

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

lst = ['angry', 'disgust', 'fear', 'happy' ,'sad', 'surprise' ,'neutral']

def get_label(pred):
    value = pred.argmax(1)
    emotion = lst[value[0]]
    print('Emotion = '+ str(emotion))
    return emotion

# Video Feed

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('/Users/harshilyadav/Desktop/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray

def start_app():
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1
        
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            roi = roi/255
            roi = roi.reshape(1,48,48,1)
            pred = model.predict(roi,verbose=1)
            output_label = get_label(pred)
            
            if output_label == 'angry':
                cv2.putText(fr, output_label, (x, y), font, 1, (255, 0, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            elif output_label == 'disgust':
                cv2.putText(fr, output_label, (x, y), font, 1, (102, 0, 51), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(102,0,51),2)
            elif output_label == 'fear':
                cv2.putText(fr, output_label, (x, y), font, 1, (153, 255, 255), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(153,255,255),2)
            elif output_label == 'surprise':
                cv2.putText(fr, output_label, (x, y), font, 1, (255, 153, 153), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,153,153),2)
            elif output_label == 'sad':
                cv2.putText(fr, output_label, (x, y), font, 1, (102, 102, 153), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(102,102,153),2)
            elif output_label == 'happy':
                cv2.putText(fr, output_label, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,255,0),2)
            elif output_label == 'neutral':
                cv2.putText(fr, output_label, (x, y), font, 1, (102, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(102,255,0),2)
            
           

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
#   model = FacialExpressionModel("model1.json", "chkPt1.h5")
    start_app()
