# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:26:09 2017

@author: 1337
"""

#read https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import time

TRAINING_MODE = False
BENCHMARK = False

model = Sequential()
model.add(  Conv2D(32,(3,3),padding='same',input_shape=(64,64,3)) )
model.add(Activation('relu'))
model.add(  Conv2D(32,(3,3),padding='same') )
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.10))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.10))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))


#custom_optimizer = keras.optimizers.rmsprop(lr = 0.001,decay=1e-4)
model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )


batch_size = 32

#https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(
        rescale=1./255,#*
        #shear_range=0.2,#random shear transformations (google)
        rotation_range=120, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False # randomly flip images
    )

training_gen = datagen.flow_from_directory(
                        'data/train',
                        target_size=(64,64),
                        batch_size = batch_size,
                        #using binary_crossentropy loss function so we need binary labels
                        class_mode='binary'
                    )

validation_gen = ImageDataGenerator(
                        rescale=1./255
                    ).flow_from_directory(
                        'data/validation',
                        target_size=(64,64),
                        batch_size = batch_size,
                        class_mode='binary'           
                    )


def get(x):
    p = model.predict(x.reshape((1,) + x.shape))
    #p = np.argmax(p)
    print(p)
    if p==[[ 0.]]:
        return "Fist"
    elif p==[[ 1.]]:
        return "Five"
    #else:
    #    return "null"

def cap():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        frame = cap.read()[1]
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        x = cv2.resize(frame,(64,64))
        inference = get(x)
        cv2.putText(frame,inference,(0,50),font,2,(255,255,255),2,cv2.LINE_AA)        
        cv2.imshow('original',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()

def k():
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()

if TRAINING_MODE:
    model.fit_generator(
                training_gen,
                steps_per_epoch=200,
                epochs=10,
                validation_data=validation_gen,
                validation_steps=100
            )
    
    # Save model and weights
    model.save("lg_deep_saved_model_001.h5")
    model.save_weights("lg_deep_saved_weights_001.h5")
    print('Saved trained model and weights')
else:
    model = keras.models.load_model("deep_saved_model_002.h5")
    model.load_weights("lg_deep_saved_weights_001.h5")
    #cap()
    


'''
*rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
'''