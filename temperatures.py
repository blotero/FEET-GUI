import numpy as np
import matplotlib.pyplot as plt
import cv2
from dermatomes import get_dermatomes

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
        return means, temp, original_temp
    else:
        mean = (temp[mask!=0]).mean()
        return mean, temp, original_temp


dic_dermatomes = {0:'Backgroud', 10:'Medial Plantar Pie Derecho', 11:'Medial Plantar Pie Izquierdo', 20:'Lateral Plantar Pie Derecho', 21:'Lateral Plantar Pie Izquierdo',
                  30:'Sural Pie Derecho', 31:'Sural Pie Izquierdo', 40:'Tibial Pie Derecho', 41:'Tibial Pie Izquierdo',
                  50:'Saphenous Pie Derecho', 51:'Saphenous Pie Izquierdo', 255:'Edges'}


derm_id = list(dic_dermatomes.keys())
derm_id.sort()
derm_names = [dic_dermatomes[key] for key in derm_id[1:-1]]


def dermatomes_temperatures(original_temp, mask):
    
    dermatomes_mask = get_dermatomes(mask.astype('uint8'))
     
    mean_temp_t_derm = np.zeros((len(derm_names)))
    
    for j, derm_id in enumerate(np.unique(dermatomes_mask)[1:-1]):
        mean_temp_t_derm[j] = original_temp[dermatomes_mask==derm_id].mean()
        
    
    return mean_temp_t_derm, dermatomes_mask