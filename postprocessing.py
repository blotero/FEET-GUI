from scipy import ndimage
import numpy as np
import cv2
from functools import partial

class PostProcessing():
    def __init__(self,  small_object_threshold):
        self.small_object_threshold = small_object_threshold
        self.default_steps = [
                    fill_inside_holes,
                    partial(opening,diameter=4),
                    partial(remove_small_objects,min_size = self.small_object_threshold),
                    partial(closing,diameter=4),
                 ]   

    def execute(self, mask):
        mask = np.squeeze(mask)
        for step in self.default_steps:
            mask = step(mask)
        return mask[...,None].astype('float32')


def  fill_inside_holes(img):
    img = img.astype('uint8')
    contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(img)
    for c in contours:
        img = cv2.drawContours(img,[c],-1,1,-1)
    return  img

def circle_structure(diameter):
    """
    ndimage.binary_opening(img, circle_structure(15))
    ndimage.binary_closing(img, circle_structure(15))
    """
    radius = diameter // 2
    x = np.arange(-radius, radius+1)
    x, y = np.meshgrid(x, x)
    r = x**2 + y**2
    return r < radius**2


def opening(img,diameter=15):
    return ndimage.binary_opening(img, circle_structure(diameter))


def closing(img,diameter=15):
    return ndimage.binary_closing(img, circle_structure(diameter))


def remove_small_objects(img, min_size=2500):
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
    connectivity=8
    img2 = np.copy(img)
    img2 = np.uint8(img2)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img2, connectivity=connectivity)
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
