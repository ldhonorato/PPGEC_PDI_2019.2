import glob
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

#TODO: Implementar outros métodos de interpolação: bilinear e bicúbica

def calc_bilinear_interpolation(img, posX, posY):
    #Get integer and fractional parts of numbers
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi+1,img.shape[0]-1)
    modYiPlusOneLim = min(modYi+1,img.shape[1]-1)

    #Get pixels in four corners
    bl = img[modYi, modXi]
    br = img[modYi, modXiPlusOneLim]
    tl = img[modYiPlusOneLim, modXi]
    tr = img[modYiPlusOneLim, modXiPlusOneLim]

    #Calculate interpolation
    b = modXf * br + (1. - modXf) * bl
    t = modXf * tr + (1. - modXf) * tl
    pxf = modYf * t + (1. - modYf) * b
    return int(pxf+0.5)

def interpolation_bilinear(original_img, newShape):
    original_shape = original_img.shape
    x_scale = original_shape[1]/newShape[1]
    y_scale = original_shape[0]/newShape[0]
    #newImage = np.zeros(newShape)

    x = np.arange(newShape[1])
    y = np.arange(newShape[0])

    x = x * x_scale
    y = y * y_scale

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, original_shape[1]-1)
    x1 = np.clip(x1, 0, original_shape[1]-1)
    y0 = np.clip(y0, 0, original_shape[0]-1)
    y1 = np.clip(y1, 0, original_shape[0]-1)

    Ia = original_img[ y0, x0 ]
    Ib = original_img[ y1, x0 ]
    Ic = original_img[ y0, x1 ]
    Id = original_img[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

    


def interpolation_NearestNeighbour(original_img, newShape):
    original_shape = original_img.shape
    x_scale = newShape[0]/original_shape[0]
    y_scale = newShape[1]/original_shape[1]
    newImage = np.zeros(newShape)
    for x in range(0,newShape[0]):
        for y in range(0,newShape[1]):
            newImage[x, y] = original_img[int(x/x_scale), int(y/y_scale)]
    return newImage

def NearestNeighbour(original_img, newShape):
    original_shape = original_img.shape
    x_scale = newShape[0]/original_shape[0]
    y_scale = newShape[1]/original_shape[1]
    newImage = np.zeros(newShape)
    for x in range(0,original_shape[0]):
        for y in range(0,original_shape[1]):
            newImage[int(x*x_scale), int(y*y_scale)] = original_img[x, y]
    return newImage

imagesTuples_path_newSize = [('n01_realce/images/zoom_in_1.png', [480, 360]),
                            ('n01_realce/images/zoom_in_2.png', [1456, 2597]),
                            ('n01_realce/images/zoom_in_3.png', [990, 720]),
                            ('n01_realce/images/zoom_out_1.png', [120, 271]),
                            ('n01_realce/images/zoom_out_2.png', [500, 317]),
                            ('n01_realce/images/zoom_out_3.png', [500, 174])]

def main():
    for imageTuple in imagesTuples_path_newSize:
        print('=================')
        imgPath = imageTuple[0] #path to im
        print('Image path: ', imgPath)
        img = imread(imgPath)
        print('Original size: ', img.shape)
        newShape = imageTuple[1] #target image size
        print('New size: ', newShape)

        insertIndex = imageName.find('.png')
        
        bilinearImage = interpolation_bilinear(img, newShape)
        plt.imshow(bilinearImage, cmap='gray')
        plt.show()

        #newImageWithoutInterpolation = NearestNeighbour(img, newShape)
        #fileName = imageName[:insertIndex] + '_outputWithoutInterpolation.png'
        #imsave(fileName, newImageWithoutInterpolation, cmap='gray')

        
        #newImageInterpolation = interpolation_NearestNeighbour(img, newShape)
        #fileName = imageName[:insertIndex] + '_outputNearestNeighbour.png'
        #imsave(fileName, newImageInterpolation, cmap='gray')
        
        #fig = plt.figure()
        #fig.add_subplot(1, 3, 1)
        #plt.imshow(img, cmap='gray')
        #fig.add_subplot(1, 3, 2)
        #plt.imshow(newImageWithoutInterpolation, cmap='gray')
        #fig.add_subplot(1, 3, 3)
        #plt.imshow(newImageInterpolation, cmap='gray')
        #plt.show()

if __name__ == "__main__":
    main()

