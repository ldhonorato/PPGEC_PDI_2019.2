import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt


def padding(img, padding_size=1, valor=0):  # caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+padding_size*2,
                      img.shape[1]+padding_size*2)) + valor
    i = (padding_size-1)
    newImg[i:img.shape[0]+i, i:img.shape[1]+i] = img

    return newImg, img.shape


'''
img é um np.array representando uma imagem 2D (MxN)
filter_kernel é um np.array (AxA) (matriz quadrada), onde A é impar (A=2a+1)
'''


def conv(img, filter_kernel):
    kernel_shape = filter_kernel.shape
    assert len(kernel_shape) == 2  # kernel é uma matriz 2D
    # filter kernel é matriz quadrada
    assert kernel_shape[0] == kernel_shape[1]
    assert kernel_shape[0] % 2 == 1  # o kernel tem dimensões impar
    assert kernel_shape[0] > 2  # kernel minimo 3x3

    padding_size = int(kernel_shape[0]//2)
    kernel_sum = np.sum(filter_kernel)
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    for i in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            newImg[i, j] = np.sum((padded_img[ni-padding_size:ni+(padding_size+1),
                                  nj-padding_size:nj+(padding_size+1)])*filter_kernel)  # sub matriz
            if kernel_sum != 0:
                newImg[i, j] = newImg[i, j] / kernel_sum

    return newImg


def statistic_filter(img, kernel_size, statistic_function):
    padding_size = kernel_size - 2
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    for i in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            subImg = padded_img[ni-padding_size:ni +
                (padding_size+1), nj-padding_size:nj+(padding_size+1)]
            newImg[i, j] = statistic_function(subImg)

    return newImg


gaussian_kernel_5x5 = np.array([[1, 4,  6,  4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4,  6,  4, 1]])

gaussian_kernel_3x3 = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

laplace_kernel_3x3 = np.array([ [0,  1, 0],
                                [1, -4, 0],
                                [0,  1, 1]])

laplace_diagonal_kernel_3x3 = np.array([ [1,  1, 1],
                                         [1, -8, 1],
                                         [1,  1, 1]])

g1 = np.array([[2, 4, 5, 4, 2],
                 [4, 9, 26, 9, 4],
                 [5, 12, 15, 12, 5],
                 [4, 9, 26, 9, 4],
                 [2, 4, 5, 4, 2]])/273
    
g2 = np.array([[1, 4, 7, 4, 1],
                 [4, 16, 26, 16, 4],
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1, 4, 7, 4, 1]])/273
    
g3 = np.array([[2, 4, 5, 4, 2],
                 [4, 9, 12, 9, 4],
                 [5, 12, 15, 12, 5],
                 [4, 9, 12, 9, 4],
                 [2, 4, 5, 4, 2]])/115

def sobel(subImg):
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    resultado = ((np.sum(gx * subImg))**2 + (np.sum(gy * subImg))**2)**0.5
    
    return resultado

def wiener(subImg):
        
    img_fft = np.fft.fft2(subImg)
    
    funcao_fft = np.fft.fft2(np.array([[1, 4, 7, 4, 1],
                 [4, 16, 26, 16, 4],
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1, 4, 7, 4, 1]])/273)
    
    resultado = img_fft*np.conjugate(funcao_fft)/(np.abs(funcao_fft) + 1)
        
    resultado = np.abs(np.fft.ifft2(resultado))
    
    return resultado[0][0]

def fWiener(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - 4):
        for j in range(img.shape[1] - 4):
            ni = i + 2
            nj = j + 2
            newImg[i, j] = wiener(img[ni-2:ni+3,nj-2:nj+3]) #sub matriz
    
    return newImg 

def fSobel(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = sobel(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg

def saveImgFromArray(imgArray, imgName):
    img = pil.fromarray(imgArray)
    img = img.convert('L')
    img.save(imgName)


def plotImages(images, nrows, ncols, title):
    i = 1
    fig = plt.figure()
    fig.suptitle(title)
    for img in images:
        ax = fig.add_subplot(nrows, ncols, i)
        plt.imshow(img[0], cmap='gray')
        plt.title(img[1])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        i += 1
    
    plt.show()

def generateGaussianKernel(sigma):
    gaussianKernel = np.zeros((3,3))
    denominador = 2*sigma**2
    for i in range(gaussianKernel.shape[0]):
        for j in range(gaussianKernel.shape[1]):
            x = i - 1
            y = j - 1
            gaussianKernel[j][i] = np.exp(-(x**2 + y**2)/denominador)
    
    return gaussianKernel