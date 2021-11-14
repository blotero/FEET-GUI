#!/usr/bin/python
# -*- coding: utf-8 -*-

#Simple code to convert grayscale image into
#rainbow colormap image

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "image", 
        help="Input image file",)

def gray_to_rainbow(img):
    #Converts grayscale image into rainbow image
    print("Converting grayscale image into rainbow image")
    input_img = plt.imread(img)
    output_filename = "rainbow_" + img_path
    plt.imsave(output_filename, input_img[:,:,0] , cmap='rainbow')
    print("Succesfully converted image")

def rainbow_to_gray(img):
    #Converts rainbow image into grayscale image
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    img_path = args.image
    print("Input image:")
    print(img_path)
    gray_to_rainbow(img_path)
    
