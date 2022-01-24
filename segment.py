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

    def extract(self):
        img_size = self.input_shape()
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
        
    def predict(self, X):
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

        return predictions
    
    def loadModel(self):
        self.interpreter = tflite.Interpreter(model_path = self.model)
        self.interpreter.allocate_tensors()

    def input_shape(self):
        input_details = self.interpreter.get_input_details()[0]['shape'][1]
        print(input_details)
        return input_details

    def whole_extract(self, dirs):
        img_size = self.input_shape()
        self.img_array=[]
        for i in range(len(dirs)):
            self.img_array.append(plt.imread(dirs[i]))
        self.img_array=np.array(self.img_array)
        self.X = np.array([cv2.resize(self.img_array[i] , (img_size , img_size), interpolation = cv2.INTER_NEAREST) for i in range(self.img_array.shape[0])])/255.
        self.Xarray  = np.array(self.X)
        self.Xarray = (self.Xarray/self.Xarray.max()).reshape(len(dirs) ,img_size , img_size , 3)
        self.Y_pred = self.predict(self.X)

    def setPath(self,im):
        self.sessionPath = im
        self.PathIsLoaded = True

    def setModel(self,model):
        self.model = model

def remove_small_objects(img, min_size=1200):
    """Remove all the objects that are smaller than a defined threshold
    Parameters
    ----------
    img : np.ndarray
        Input image to clean
    min_size : int, optional
        Threshold to be used to remove all smaller objects, by default 1200
    Returns
    -------
    np.ndarray
        Cleaned image
    """
    img2 = np.copy(img)
    img2 = np.uint8(img2)
    nb_components, output, stats, centroids = connectedComponentsWithStats(img2, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2 
