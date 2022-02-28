#Polygon segmentation using pretrained model with U-Net
#AI model avaliable in https://gihthub.com/Rmejiaz/Feet_U-Net

import numpy as np
import os
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from cv2 import connectedComponentsWithStats
import cv2




class ImageToSegment():
    def __init__(self):
        self.thereIsX = False
        self.X = None
        self.Xarray = None
        self.imageIsLoaded = False
        self.model = None

    def predict(self, X):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = np.float32(X)

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        self.interpreter.invoke()  # predict

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        
        return output_data

    def loadModel(self):
        self.interpreter = tflite.Interpreter(model_path = self.model)
        self.interpreter.allocate_tensors()

    def input_shape(self):
        input_details = self.interpreter.get_input_details()[0]['shape'][1]
        print(input_details)
        return input_details

    def extract(self, cmap = 'rainbow'):
        img_size = self.input_shape() # Input shape of the cnn
        if cmap == 'Hierro':  # If cmap is rainbow, convert to grayscale
            self.img = plt.imread(self.imPath)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            new_img = np.empty((self.img.shape[0], self.img.shape[1], 3))
            new_img[:,:,0] = new_img[:,:,1] = new_img[:,:,2] = self.img # Add three channels to be compatible with dl models
            self.img = new_img
        
        if cmap == 'Gris':
            self.img = plt.imread(self.imPath)

        # self.X = tf.convert_to_tensor(self.img)
        self.X = self.img
        self.X = cv2.resize(self.X, (img_size, img_size), interpolation = cv2.INTER_NEAREST)
        self.Xarray  = np.array(self.X)
        self.Xarray = (self.Xarray/self.Xarray.max()).reshape(img_size , img_size , 3)
        self.X = np.expand_dims(self.X,0)/255.
        self.Y_pred = self.predict(self.X)

    def setPath(self,im):
        self.imPath = im
        self.imageIsLoaded = True

    def setModel(self,model):
        self.model = model
        
        
class SessionToSegment():
    def __init__(self):
        self.thereIsX = False
        self.X = None
        self.Xarray = None
        
    def predict(self, X, progressBar):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        predictions = []

        for i in range(X.shape[0]):
            input_shape = input_details[0]['shape']
            input_data = np.float32(X[i])
            input_data = np.expand_dims(input_data, axis=0)
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)

            self.interpreter.invoke()  # predict

            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data)
            progressBar.setValue((100*i+1)/X.shape[0])

        return predictions
    
    def loadModel(self):
        self.interpreter = tflite.Interpreter(model_path = self.model)
        self.interpreter.allocate_tensors()

    def input_shape(self):
        input_details = self.interpreter.get_input_details()[0]['shape'][1]
        print(input_details)
        return input_details

    def whole_extract(self, dirs, cmap = 'rainbow',progressBar=None):
        img_size = self.input_shape()
        self.img_array=[]
        if cmap == 'Gris':
            for i in range(len(dirs)):
                self.img_array.append(plt.imread(dirs[i]))

        
        if cmap == 'Hierro': # If input images are in gnuplot2, convert them to grayscale
            for i in range(len(dirs)):
                img = plt.imread(dirs[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                new_img = np.empty((img.shape[0], img.shape[1], 3)) # Add three more channels to the image
                new_img[:,:,0] = new_img[:,:,1] = new_img[:,:,2] = img
                img = new_img

                self.img_array.append(img)

        self.img_array=np.array(self.img_array)
        self.X = np.array([cv2.resize(self.img_array[i] , (img_size , img_size), interpolation = cv2.INTER_NEAREST) for i in range(self.img_array.shape[0])])/255.
        self.Xarray  = np.array(self.X)
        self.Xarray = (self.Xarray/self.Xarray.max()).reshape(len(dirs) ,img_size , img_size , 3)
        self.Y_pred = self.predict(self.X,progressBar)

    def setPath(self,im):
        self.sessionPath = im
        self.PathIsLoaded = True

    def setModel(self,model):
        self.model = model

