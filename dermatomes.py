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


def get_centroids_and_tby(img):
    """Get centroids and top-bottom y for initialization template of dermatomes
    """

    _, thresh = cv2.threshold(img,127,255,0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = list(contours)
    contours.sort(reverse=True,key= lambda c : cv2.contourArea(c))
    centroids = []

    for i,c in enumerate(contours[:2]):
       M = cv2.moments(c)

       # calculate x,y coordinate of center
       cX = int(M["m10"] / M["m00"])
       cY = int(M["m01"] / M["m00"])
       #y Top and y Bottom
       yTop = c[c[:, :, 1].argmin()][0][1]
       yBot = c[c[:, :, 1].argmax()][0][1]


       centroids.append([cX,cY,yTop,yBot])

    centroids.sort(key = lambda x: x[0])
    return centroids

def plot_predict(y,y_pred):
    red  = np.logical_and(~y,y_pred)[...,None]  #false positive
    green = np.logical_and(y,y_pred)[...,None]  #true positive
    blue  = np.logical_and(y,~y_pred)[...,None]  #false negative
    image =  np.concatenate((red,green,blue),axis=2).astype('float')
    return image


def reshape(hight,foot_image):
    width = int(hight*foot_image.shape[1]/foot_image.shape[0])
    width = width + width%2
    hight = hight + hight%2
    return cv2.resize(foot_image, (width,hight), interpolation = cv2.INTER_NEAREST)


def put_image_in_point(img1,img2,point):
    shape = img2.shape
    img1[point[1]-shape[0]//2:point[1]+shape[0]//2,point[0]-shape[1]//2:point[0]+shape[1]//2] = img2
    return img1

def make_moving_image(right_foot,left_foot,final_shape,centroids_and_tby):
    moved = np.zeros(shape=(final_shape))
    
    hight_right =  centroids_and_tby[0][3] -centroids_and_tby[0][2]
    right_foot = reshape(hight_right,right_foot)

    left_right =  centroids_and_tby[1][3] -centroids_and_tby[1][2]
    left_foot = reshape(left_right,left_foot)

    moved = put_image_in_point(moved,right_foot,centroids_and_tby[0][:2])
    moved = put_image_in_point(moved,left_foot,centroids_and_tby[1][:2])
    return moved

def no_rigid_registration(fixed_image, moving_image): 
    fixed_image =  sitk.Cast(sitk.GetImageFromArray(fixed_image.copy()),sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_image.copy()),sitk.sitkFloat32)

    transformDomainMeshSize=[3]*fixed_image.GetDimension()

    tx = sitk.BSplineTransformInitializer(fixed_image,
                                      transformDomainMeshSize)   

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsGradientDescentLineSearch(learningRate=10.,
                                              numberOfIterations=100,
                                              convergenceMinimumValue=1e-10,
                                              convergenceWindowSize=10)

    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkNearestNeighbor)#sitk.sitkLinear)#

    R.SetInitialTransformAsBSpline(tx,
                                   inPlace=False,
                                   scaleFactors=[1,2,4,8])
    R.SetShrinkFactorsPerLevel([4,2,1])
    R.SetSmoothingSigmasPerLevel([4,2,1])

    outTx = R.Execute(fixed_image, moving_image)
    return outTx

def resample(moving_image,fixed_image,registration_transform):
    fixed_image =  sitk.Cast(sitk.GetImageFromArray(fixed_image),sitk.sitkFloat32)
    moving_image = sitk.Cast(sitk.GetImageFromArray(moving_image),sitk.sitkFloat32)
    return sitk.GetArrayFromImage(sitk.Resample(moving_image,fixed_image, registration_transform,sitk.sitkNearestNeighbor))


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
    fixed_image = fixed_image[...,0]
    
    right_foot = cv2.flip(cv2.imread(path_right_foot)[...,2],1)

    left_foot = cv2.imread(path_left_foot)[...,2] 
    left_foot[(left_foot!=0)&(left_foot!=255)] = left_foot[(left_foot!=0)&(left_foot!=255)] + 1 

    centroids_and_tby = get_centroids_and_tby(fixed_image)
    moving_image = make_moving_image(right_foot,left_foot,fixed_image.shape,centroids_and_tby)
    moving_image_masked = (moving_image > 0).astype('int8')

    registration_transform = no_rigid_registration(fixed_image,moving_image_masked ) 

    output_dermatomes = resample(moving_image,fixed_image,registration_transform)

    return output_dermatomes



def main(args):
    path_image = args['IMG_PATH']
    path_mask = args['MASK_PATH']

    mask = cv2.imread(path_mask)
    img = cv2.imread(path_image)
    dermatomes = get_dermatomes(mask)

    plt.figure(figsize=(20,10))

    plt.subplot(141)
    plt.imshow(mask[...,0])

    plt.subplot(142)
    plt.imshow(dermatomes)

    plt.subplot(143)
    plt.imshow(plot_predict(mask[...,0],dermatomes>0))

    plt.subplot(144)
    img[dermatomes==255,0] = 255
    img[dermatomes==255,1] = 0
    img[dermatomes==255,2] = 0
    plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    main(args)