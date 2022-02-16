"""
Foot Dermatomes 
Usage:
    dermatomes.py IMG_PATH MASK_PATH

Options:
    IMG_PATH    Path to thermographic image
    MASK_PATH   Path segmentation mask
"""
import docopt


import cv2
import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
import time 


def plot_predict(y,y_pred):
    red  = np.logical_and(~y,y_pred)[...,None]  #false positive
    green = np.logical_and(y,y_pred)[...,None]  #true positive
    blue  = np.logical_and(y,~y_pred)[...,None]  #false negative
    image =  np.concatenate((red,green,blue),axis=2).astype('float')
    return image


def refine_countour(dermatomes):
    without_countours = dermatomes.copy()
    without_countours[dermatomes==255] = 0
    uniques = np.unique(without_countours)

    for unique in uniques:
        binary_img = (without_countours==unique).astype('uint8')
        contours, _ = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        dermatomes = cv2.drawContours(dermatomes,contours,-1,255,1)

    return dermatomes


def no_rigid_registration(fixed_image, moving_image): 
    fixed_image =  sitk.Cast(sitk.GetImageFromArray(fixed_image.copy()),sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_image.copy()),sitk.sitkFloat32)

    transformDomainMeshSize=[3]*fixed_image.GetDimension()

    tx = sitk.BSplineTransformInitializer(fixed_image,
                                      transformDomainMeshSize)   

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsGradientDescentLineSearch(learningRate=10.,
                                              numberOfIterations=600,
                                              convergenceMinimumValue=1e-20,
                                              convergenceWindowSize=30)

    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.2,seed=42)
    R.SetInterpolator(sitk.sitkNearestNeighbor)#sitk.sitkLinear)#

    R.SetInitialTransformAsBSpline(tx,
                                   inPlace=False,
                                   scaleFactors=[1,2,4,8])
    #R.SetShrinkFactorsPerLevel([4,2,1])
    #R.SetSmoothingSigmasPerLevel([4,2,1])

    outTx = R.Execute(fixed_image, moving_image)
    return outTx

def resample(moving_image,fixed_image,registration_transform):
    fixed_image =  sitk.Cast(sitk.GetImageFromArray(fixed_image),sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_image),sitk.sitkFloat32) 
    return sitk.GetArrayFromImage(sitk.Resample(moving_image,fixed_image, registration_transform,sitk.sitkNearestNeighbor))



def register_one_foot(foot,dermatomes):
    hight = foot.shape[0]
    width = foot.shape[1]
    dermatomes = cv2.resize(dermatomes, (width,hight), interpolation = cv2.INTER_NEAREST)
    mask_dermatomes = (dermatomes.copy() >0).astype('float')
    registration_transform = no_rigid_registration(foot,mask_dermatomes) 
    registered = resample(dermatomes,foot,registration_transform)
    return  registered

    

def extract_feet(img):
    """Get centroids and top-bottom y for initialization template of dermatomes
    """

    img = img.astype('uint8')
    contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = list(contours)
    contours.sort(reverse=True,key= lambda c : cv2.contourArea(c))
    coord = []

    for i,c in enumerate(contours[:2]):

       #y Top and y Bottom
       yTop = c[c[:, :, 1].argmin()][0][1] - 5
       yBot = c[c[:, :, 1].argmax()][0][1] + 5
       xRig = c[c[:, :, 0].argmin()][0][0] - 5
       xLef = c[c[:, :, 0].argmax()][0][0] + 5
       coord.append([yTop,yBot,xRig,xLef,c])

       
    coord.sort(key = lambda x: x[2]) #order foots

    right_foot = np.zeros_like(img)
    right_foot = cv2.drawContours(right_foot,[coord[0][-1]],-1,1,-1)
    right_foot = right_foot[coord[0][0]:coord[0][1],coord[0][2]:coord[0][3]]

    left_foot = np.zeros_like(img)
    left_foot = cv2.drawContours(left_foot,[coord[1][-1]],-1,1,-1)
    left_foot = left_foot[coord[1][0]:coord[1][1],coord[1][2]:coord[1][3]]

    return right_foot, left_foot,coord



def get_dermatomes(fixed_image,path_right_foot='images/dermatomes.png',path_left_foot='images/dermatomes.png'):
    """
    0 -> background
    255 -> boundary

    right-left 
       10-11 -> Medial Plantar
       20-21 -> Lateral Plantar
       30-31 -> Sural
       40-41 -> Tibial
       50-51 -> Saphenous
    """
    #all in hxw

    fixed_image = np.squeeze(fixed_image)

    right_dermatomes = cv2.flip(cv2.imread(path_right_foot)[...,2],1)

    left_dermatomes = cv2.imread(path_left_foot)[...,2] 
    left_dermatomes[(left_dermatomes!=0)&(left_dermatomes!=255)] = left_dermatomes[(left_dermatomes!=0)&(left_dermatomes!=255)] + 1 


    right_foot,left_foot, coord = extract_feet(fixed_image)
    
    right_dermatomes = register_one_foot(right_foot,right_dermatomes)
    left_dermatomes = register_one_foot(left_foot,left_dermatomes)

    output_dermatomes = np.zeros_like(fixed_image,dtype='float')
    output_dermatomes[coord[0][0]:coord[0][1],coord[0][2]:coord[0][3]] = right_dermatomes
    output_dermatomes[coord[1][0]:coord[1][1],coord[1][2]:coord[1][3]] = left_dermatomes

    output_dermatomes =  refine_countour(output_dermatomes)
    return output_dermatomes



def main(args):
    path_image = args['IMG_PATH']
    path_mask = args['MASK_PATH']

    mask = cv2.imread(path_mask) 
    mask = cv2.resize(mask,(224,224),interpolation=cv2.INTER_NEAREST)
    mask = mask[...,0] != 0
    
    img = cv2.imread(path_image)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
    
    t1 = time.time()
    dermatomes = get_dermatomes(mask)
    tf = time.time()-t1
    print(f'Time : {tf:.4f}')

    right_foot,left_foot, _ = extract_feet(mask)
    plt.figure(figsize=(20,10))

    plt.subplot(241)
    plt.imshow(mask)
    
    plt.subplot(242)
    plt.imshow(right_foot)

    plt.subplot(243)
    plt.imshow(left_foot)

    plt.subplot(244)
    plt.imshow(dermatomes)
    
    plt.subplot(245)
    plt.imshow(plot_predict(mask,dermatomes>0))

    plt.subplot(246)
    img[dermatomes==255,0] = 255
    img[dermatomes==255,1] = 0
    img[dermatomes==255,2] = 0
    plt.imshow(img)


    plt.show()
#

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    main(args)