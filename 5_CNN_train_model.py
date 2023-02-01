# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 21:21:29 2022
@author: cheng
"""

# Deep Learning CNN model to recognize face

# TrainingImagePath='C:/Users/cheng/Documents/Syracuse/IST718/Face_2/Training_Data/'
# TestingImagePath='C:/Users/cheng/Documents/Syracuse/IST718/Face_2/Testing_Data/'
TrainingImagePath = 'C:/Users/cheng/Documents/Syracuse/IST718/Face_2/final_train_data/'
TestingImagePath = 'C:/Users/cheng/Documents/Syracuse/IST718/Face_2/final_test_data/'

from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

'''############ Generating test/train sets ############'''
# These hyper parameters helps to generate slightly twisted versions of images to help training
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(330, 330),
        batch_size=32,
        class_mode='categorical')

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
        TestingImagePath,
        target_size=(330, 330),
        batch_size=32,
        class_mode='categorical')

# Printing class labels for each face
test_set.class_indices
	
'''############ Creating lookup table for all faces ############'''
# class_indices have the numeric tag for each face
TrainClasses=training_set.class_indices
 
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
 
# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID",ResultMap)
 
# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

'''######################## Create CNN deep learning model ########################'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''Initializing the Convolutional Neural Network'''
classifier= Sequential()

''' STEP--1 Convolution '''
# Adding the first layer of CNN
# we are using the format (64,64,3) RGB because we are using TensorFlow backend
batch = 32
classifier.add(Convolution2D(batch, kernel_size=(5, 5), strides=(1, 1), input_shape=(330,330,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# ADDITIONAL LAYER of CONVOLUTION for better accuracy
classifier.add(Convolution2D(batch, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

import time

# Measuring the time taken by the model to train
StartTime=time.time()

# Starting the model training
history = classifier.fit(
                    training_set,
                    steps_per_epoch=32,
                    epochs=25,
                    validation_data=test_set,
                    shuffle=True)

EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')

predictions = classifier.predict(test_set2)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)

from sklearn.metrics import classification_report # print classification accuracy
test_set2 = test_datagen.flow_from_directory(
        TestingImagePath,
        target_size=(330, 330),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

class_labels = list(test_set2.class_indices.keys())  
true_classes = test_set2.classes
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

import pandas

df = pandas.DataFrame(report).transpose()
df.to_csv('C:/Users/cheng/Documents/Syracuse/IST718/Face_2/class_report.csv')