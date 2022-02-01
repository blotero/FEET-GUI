import numpy as np
import matplotlib.pyplot as plt

def mean_temperature(image , mask , range_=[22.5 , 35.5], plot = False):
    """Get mean temperature of feet image based on mask and scale
    Parameters
    ----------
    image: np.ndarray, RGB input image as numpyarray
    mask: np.ndarray, probability mask as output of segmentation. Must have same dimensions as input image
    range_: list, temperature scales in Celsius [min, max]
    plot: boolean, wheter a figure is shown or not
    """
    temp = image*(range_[1] - range_[0]) + range_[0]
    print(np.unique(temp))
    temp*=mask
    mean = (temp[temp!=0]).mean()
    print(np.unique(temp))

    print(mean)
    if plot:
        plt.figure()
        plt.imshow(temp,norm=None, cmap='gray')
        print(np.unique(temp))
        plt.colorbar(ticks=np.linspace(range_[0]  , range_[1] , 10 )  )
        plt.axis("off")
        #plt.clim(range_[0] , range_[1])
        plt.show()

    return mean, temp
