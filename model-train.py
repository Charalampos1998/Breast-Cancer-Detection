from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import time

NAME = "mams-cnn-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)


train_dataset = train.flow_from_directory('C:/Users/mpamp/Desktop/br-canc/data/train',
                                            target_size=(200,200),
                                            batch_size= 3,
                                            class_mode= 'binary')

validation_dataset = train.flow_from_directory('C:/Users/mpamp/Desktop/br-canc/data/valid',
                                            target_size=(200,200),
                                            batch_size=3,
                                            class_mode='binary')

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                      tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                      tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                     tf.keras.layers.Flatten(),
                                     #
                                     tf.keras.layers.Dense(512,activation='relu'),
                                     #
                                     tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
            steps_per_epoch = 5,
            epochs = 150,
            validation_data=validation_dataset,
            callbacks=[tensorboard])

model.save('first.model')




