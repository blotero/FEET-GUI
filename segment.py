#Polygon segmentation using pretrained model with U-Net
#AI model avaliable in https://gihthub.com/Rmejiaz/Feet_U-Net

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
class ImageToSegment():
    def __init__(self):
        self.thereIsX = False
        self.X = None
        self.Xarray = None
        self.imageIsLoaded = False

    def extract(self):
        img_size = 224
        self.img = plt.imread(self.imPath)/255
        self.X = tf.convert_to_tensor(self.img)
        self.X = tf.image.resize(self.X , (img_size , img_size))
        self.Xarray  = np.array(self.X)
        self.Xarray = (self.Xarray/self.Xarray.max()).reshape(img_size , img_size , 3)
        self.X = tf.expand_dims(self.X,0)
        self.model = tf.keras.models.load_model('Model1.h5')
        self.Y_pred = self.model.predict(self.X)
        
    def setPath(self,im):
        self.imPath = im
        self.imageIsLoaded = True
        
        
class SessionToSegment():
    def __init__(self):
        self.thereIsX = False
        self.X = None
        self.Xarray = None
        
    def whole_extract(self, dirs):
        img_size = 224
        self.img_array=[]
        for i in range(len(dirs)):
            self.img_array.append(plt.imread(dirs[i])/255)
        self.img_array=np.array(self.img_array)
        self.X = tf.convert_to_tensor(self.img_array)
        self.X = tf.image.resize(self.X , (img_size , img_size))
        self.Xarray  = np.array(self.X)
        self.Xarray = (self.Xarray/self.Xarray.max()).reshape(len(dirs) ,img_size , img_size , 3)
        self.model = tf.keras.models.load_model('Model1.h5')
        self.Y_pred = self.model.predict(self.X)
        print(self.Y_pred)

    def setPath(self,im):
        self.sessionPath = im
        self.PathIsLoaded = True
