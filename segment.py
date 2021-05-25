#Polygon segmentation using pretrained model with U-Net
#Avaliable in https://gihthub.com/Rmejiaz/Feet_U-Net

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
class Image_2_seg():
    def __init__(self):
        self.thereIsX = False
        self.X = None

    def extract(self):
        img_size = 224
        self.img = plt.imread(self.imPath)/255
        self.X = tf.convert_to_tensor(self.img)
        self.X = tf.image.resize(self.X , (img_size , img_size))
        self.X = tf.expand_dims(self.X,0)
        self.model = tf.keras.models.load_model('Model1.h5')
        self.Y_pred = self.model.predict(self.X)

    def setPath(self,im):
        self.imPath = im
        self.imageIsLoaded = True
