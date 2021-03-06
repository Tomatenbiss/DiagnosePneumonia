''' Code inspired by https://www.kaggle.com/faizunnabi/diagnose-pneumonia/notebook'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D,AveragePooling2D, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau




''' Model hyperparameters such as window size, batch size for training and number of epochs'''
image_height = 32
image_width = 32
batch_size = 32
no_of_epochs  = 300

''' Architecture of the Model '''
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(image_height,image_width,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(units=361*8,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

''' Print a summary of the model '''
model.summary()

''' Declare the image data generator for batch creation '''
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

''' Generate training dataset '''
training_set = train_datagen.flow_from_directory('../chest_xray/train',
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

''' Generate test dataset '''
test_set = test_datagen.flow_from_directory('../chest_xray/test',
                                            target_size=(image_width, image_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

''' Callback function to reduce learning rate by factor 0.1 when a plateau is reached '''
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)
''' Reduce learning rate when a plateu '''
callbacks = [reduce_learning_rate]

''' Train the model '''
history = model.fit_generator(training_set,
                    steps_per_epoch=5216//batch_size,
                    epochs=no_of_epochs,
                    validation_data=test_set,
                    validation_steps=624//batch_size,
                    callbacks=callbacks
                   )

''' Store model '''
model.save('own_architecture_with_dropout.dat')