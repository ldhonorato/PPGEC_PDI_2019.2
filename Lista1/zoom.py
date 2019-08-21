import glob
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

#TODO: Implementar outros métodos de interpolação: bicúbica
def cubic_interpolation(p, x):
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

def interpolation_bicubic(original_img, newShape):
    original_shape = original_img.shape
    x_scale = original_shape[1]/newShape[1]
    y_scale = original_shape[0]/newShape[0]
    newImage = np.zeros(newShape)
    #for i_x in range(newShape[1]):
    #    for i_y in range(newShape[0]):
            
    return newImage

def interpolation_bilinear(original_img, newShape):
    original_shape = original_img.shape
    x_scale = original_shape[1]/newShape[1]
    y_scale = original_shape[0]/newShape[0]
    newImage = np.zeros(newShape)

    for i_x in range(newShape[1]):
        for i_y in range(newShape[0]):
            #position in the original image
            x = i_x * x_scale
            y = i_y * y_scale

            x1 = int(x)
            y1 = int(y)
            x2 = min(x1+1,original_img.shape[1]-1)
            y2 = min(y1+1,original_img.shape[0]-1)

            q11 = original_img[y1, x1]
            q12 = original_img[y2, x1]
            q21 = original_img[y1, x2]
            q22 = original_img[y2, x2]

            if x1 != x2: #to avoid zero division
                r1 = q11*((x2-x)/(x2-x1)) + q21*((x-x1)/(x2-x1))
                r2 = q12*((x2-x)/(x2-x1)) + q22*((x-x1)/(x2-x1))
            else:
                r1 = q12 #in fact q12 = q22 = r1 = r2
                r2 = q22

            if y1 != y2: #to avoid zero division
                p = r1*((y2-y)/(y2-y1)) + r2*((y-y1)/(y2-y1))
            else:
                p = r1 #if y1 = y2 then r1 = r2
            newImage[i_y, i_x] = p
    return newImage


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

        imageName = imgPath.split('/')[-1]
        insertIndex = imageName.find('.png')
        
        bilinearImage = interpolation_bilinear(img, newShape)
        plt.imshow(bilinearImage, cmap='gray')
        plt.show()
        fileName = imageName[:insertIndex] + '_outputBilinearInterpolation.png'
        imsave(fileName, bilinearImage, cmap='gray')

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

