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

'''
TRAINING_MODE = 1
RUN_MODE = 2
TRAIN_CHKPT = 3
EVAL = 4
init funcs = 0
'''
MODE = 0

p=None

BENCHMARK = False
states = ["Fist",
          "Five",
          "index",
          "md",
          "null",
          "peace"
         ]

model = Sequential()
model.add(  Conv2D(16,(3,3),padding='same',input_shape=(64,64,1)) )
model.add(Activation('relu'))
model.add(  Conv2D(16,(3,3),padding='same') )
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.10))

model.add(Conv2D(32,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3), padding='same'))
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
model.add(Dropout(0.1))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(6))
model.add(Activation('softmax'))




#custom_optimizer = keras.optimizers.rmsprop(lr = 0.001,decay=1e-4)
model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )


batch_size = 128

#https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(
        rescale=1.0/255,#*
        #shear_range=0.2,#random shear transformations (google)
        rotation_range=60, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False # randomly flip images
    )

training_gen = datagen.flow_from_directory(
                        'data/train',
                        color_mode='grayscale',
                        target_size=(64,64),
                        batch_size = batch_size,
                        #using binary_crossentropy loss function so we need binary labels
                        class_mode="categorical",
                        shuffle=True
                    )

validation_gen = ImageDataGenerator(
                        rescale=1.0/255
                    ).flow_from_directory(
                        'data/validation',
                        color_mode='grayscale',
                        target_size=(64,64),
                        batch_size = batch_size,
                        class_mode='categorical', 
                        shuffle=True
                    )

def load_weights():
    #global model
    #model = keras.models.load_model("newMultiM_001.h5")
    #print("model loaded")
    model.load_weights("newMultiW_001.h5")
    print("weights loaded")

def save_weights():
    # Save model and weights
    model.save("newMultiM_001.h5")
    model.save_weights("newMultiW_001.h5")
    print('Saved trained model and weights')


def train():
    model.fit_generator(
                training_gen,
                steps_per_epoch=training_gen.samples // batch_size,
                epochs=20,
                validation_data=validation_gen,
                validation_steps=validation_gen.samples // batch_size
            )
    eval()
    save_weights()

def eval():
    scores = model.evaluate_generator(
            validation_gen,
            steps=validation_gen.samples // batch_size
        )
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def get(x):
    global p
    p = model.predict(x.reshape((1,) + x.shape))
    q = np.argmax(p)
    print(p*100//1)
    return states[q]

frame = []

def start_cap():
    global frame
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    c=0
    print("camera loaded")
    while True:
        frame = cap.read()[1]
        frame = cv2.flip(frame,1)
        x = frame[:256,384:]
        cv2.rectangle(frame, (639,0),(384,256),(0,255,0),2)
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x,(64,64))
        if(c):
            print("camera init complete")
        #Don't ever forget normalization idiot
        inference = get(x/255)
        if(c):
            print("inference functional")
            c=1
        cv2.putText(frame,inference,(0,50),font,2,(255,255,255),2,cv2.LINE_AA)        
        cv2.imshow('original',frame)
        cv2.imshow('read',x)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()

def k():
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()

    
def load_and_train():
    load_weights()
    train()
    
def load_and_cap():
    load_weights()
    start_cap()

if __name__ == '__main__':
    if MODE==1:
        train()
    elif MODE==2:
        load_and_cap()
    elif MODE==3:
        load_and_train()
    elif MODE==4:
        eval()
    
       


'''
*rescale is a value by which we will multiply the data before any other
 processing. Our original images consist in RGB coefficients in the 0-255,
 but such values would be too high for our models to process (given a typical
 learning rate), so we target values between 0 and 1 instead by scaling with
 a 1/255. factor.
'''