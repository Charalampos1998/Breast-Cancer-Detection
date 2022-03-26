from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

dir_path = 'C:/Users/mpamp/Desktop/br-canc/data/test'

model = tf.keras.models.load_model("first.model")


for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '/'  + i,target_size=(200,200))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis =0)
    images = np.vstack([x])
    val = model.predict(images)
    if val == 0:
        print("bening")
    else:
        print("malignat")
