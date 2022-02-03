import numpy as np
import matplotlib.pyplot as plt
import cv2

def mean_temperature(image , mask , range_=[22.5 , 35.5], plot = False):
    """Get mean temperature of feet image based on mask and scale
    Parameters
    ----------
    image: np.ndarray, RGB input image as numpyarray
    mask: np.ndarray, probability mask as output of segmentation. Must have same dimensions as input image
    range_: list, temperature scales in Celsius [min, max]
    plot: boolean, wheter a figure is shown or not
    """
    original_temp = image*(range_[1] - range_[0]) + range_[0]
    #print(np.unique(temp))
    #print(f"Dimensiones mask : {mask.shape}")
    temp = original_temp * mask       
    if plot:
        plt.figure()
        plt.imshow(temp,norm=None, cmap='gray')
        print(np.unique(temp))
        plt.colorbar(ticks=np.linspace(range_[0]  , range_[1] , 10 )  )
        plt.axis("off")
        #plt.clim(range_[0] , range_[1])
        plt.show()

    result = cv2.connectedComponentsWithStats(mask.astype('uint8'))    

    if result[0] == 3:
        #Find left and right feet masks
        right_mask = np.where(result[1] == 1, 1, 0)
        left_mask = np.where(result[1] == 2, 1, 0)
        #Map with their temperatures
        right_temp = right_mask * original_temp
        left_temp = left_mask * original_temp
        #Get final mean values
        left_mean = (left_temp[left_mask!=0]).mean()
        right_mean = (right_temp[right_mask!=0]).mean()
        means = [left_mean, right_mean]        
        return means, temp
    else:
        mean = (temp[mask!=0]).mean()
        return mean, temp
