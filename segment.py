#Acá se insertará el código capaz de segmentar los pies en polígonos
import numpy as np
import os


class Image_2_seg():
    def __init__(self):
        self.nothing=None
    def extract(self):
        self.segments=np.random.randint(0,100,size=(2,2))
        self.bin_img=np.random.randint(0,2,size=(1000))
