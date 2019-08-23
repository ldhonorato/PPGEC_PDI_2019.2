import glob
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import math
#=================================================================
#Bicubic interpolation - Catmull-Rom spline approach (https://www.paulinternet.nl/?page=bicubic)
#f(p_0,p_1,p_2,p_3,x) = (-\tfrac{1}{2}p_0 + \tfrac{3}{2}p_1 - \tfrac{3}{2}p_2 + \tfrac{1}{2}p_3)x^3 + (p_0 - \tfrac{5}{2}p_1 + 2p_2 - \tfrac{1}{2}p_3)x^2 + (-\tfrac{1}{2}p_0 + \tfrac{1}{2}p_2)x + p_1
'''
def cubic_interpolation(p, x):
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

def bicubic_interpolation(p, x, y):
    #g(x,y) = f(f(p_{00},p_{01},p_{02},p_{03},y), f(p_{10},p_{11},p_{12},p_{13},y), f(p_{20},p_{21},p_{22},p_{23},y), f(p_{30},p_{31},p_{32},p_{33},y),x)
    f = []
    for j in range(4):
        f.append(cubic_interpolation(p[:,j], y))
    
    return cubic_interpolation(f, x)

def interpolation_bicubic(original_img, newShape):
    original_shape = original_img.shape
    x_scale = original_shape[1]/newShape[1]
    y_scale = original_shape[0]/newShape[0]
    newImage = np.zeros(newShape)

    original_img_padded = np.zeros((original_shape[0] + 4, original_shape[1] + 4))
    original_img_padded[2:original_shape[0]+2,2:original_shape[1]+2] = original_img
    
    for i_x in range(newShape[1]):
        for i_y in range(newShape[0]):
            x = i_x * x_scale
            y = i_y * y_scale

            x0 = int(x)
            y0 = int(y)
            p = np.zeros((4,4))
            p = original_img_padded[y0:y0+4, x0:x0+4]
            #for l in range(4):
            #    for k in rangeimport sys(4):
            #        p[l,k] = original_img_padded[y0 + l, x0 + k]
            newImage[i_y, i_x] = bicubic_interpolation(p, x, y)
        sys.stdout.write('\r')
        sys.stdout.write("%d %%" % (int(i_x/newShape[1]*100)))
    return newImage
'''
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

def padding(img,H,W):
    zimg = np.zeros((H+4,W+4))
    zimg[2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2] = img[:,0:1]
    zimg[H+2:H+4,2:W+2] = img[H-1:H,:]
    zimg[2:H+2,W+2:W+4] = img[:,W-1:W]
    zimg[0:2,2:W+2] = img[0:1,:]
    #Pad the missing eight points
    zimg[0:2,0:2] = img[0,0]
    zimg[H+2:H+4,0:2] = img[H-1,0]
    zimg[H+2:H+4,W+2:W+4] = img[H-1,W-1]
    zimg[0:2,W+2:W+4] = img[0,W-1]
    
    return zimg

def bicubic(img, imgshape, a=-0.5):
    #Get image size
    width = imgshape[1]
    height = imgshape[0]
    H,W = img.shape
    
    img = padding(img,H,W)
    
    ratioH = height/H
    ratioW = width/W
    
    #Create new image
    dH = math.floor(height)
    dW = math.floor(width)
    im = np.zeros((dH, dW))

    hx = 1/ratioW
    hy = 1/ratioH

    for j in range(dH):
        for i in range(dW):
            x, y = i * hx + 2 , j * hy + 2

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
            mat_m = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                               [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                               [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                               [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])
            mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
            im[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r)
            
            #print("Estou em (i, j) = " + str((i, j)))
    return im
    #img = pil.fromarray(im)
    #img = img.convert("L")
    #img.save(nome + ".png")

#=================================================================
#Bilinear Interpolation
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

imagesTuples_path_newSize = [('images/zoom_in_1.png', [480, 360]),
                            ('images/zoom_in_2.png', [1456, 2597]),
                            ('images/zoom_in_3.png', [990, 720]),
                            ('images/zoom_out_1.png', [120, 271]),
                            ('images/zoom_out_2.png', [500, 317]),
                            ('images/zoom_out_3.png', [500, 174])]

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
        
        bicubicImage = bicubic(img, newShape)
        fileName = imageName[:insertIndex] + '_bicubic.png'
        imsave(fileName, bicubicImage, cmap='gray')

        bilinearImage = interpolation_bilinear(img, newShape)
        fileName = imageName[:insertIndex] + '_bilinear.png'
        imsave(fileName, bilinearImage, cmap='gray')

        newImageInterpolation = interpolation_NearestNeighbour(img, newShape)
        fileName = imageName[:insertIndex] + '_NearestNeighbour.png'
        imsave(fileName, newImageInterpolation, cmap='gray')

        #newImageWithoutInterpolation = NearestNeighbour(img, newShape)
        #fileName = imageName[:insertIndex] + '_outputWithoutInterpolation.png'
        #imsave(fileName, newImageWithoutInterpolation, cmap='gray')
        
        #plt.imshow(bicubicImage, cmap='gray')
        #plt.show()
        #plt.imshow(bilinearImage, cmap='gray')
        #plt.show()
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

