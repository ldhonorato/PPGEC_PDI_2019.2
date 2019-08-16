import glob
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

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
                            ('n01_realce/images/zoom_in_3.png', [990, 720])]

for imageTuple in imagesTuples_path_newSize:
    print('=================')
    imgPath = imageTuple[0]
    print('Image path: ', imgPath)
    img = imread(imgPath)
    print('Original size: ', img.shape)
    newShape = imageTuple[1]
    print('New size: ', newShape)
    newImageWithoutInterpolation = NearestNeighbour(img, newShape)
    newImageInterpolation = interpolation_NearestNeighbour(img, newShape)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(newImageWithoutInterpolation, cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(newImageInterpolation, cmap='gray')
    plt.show()

