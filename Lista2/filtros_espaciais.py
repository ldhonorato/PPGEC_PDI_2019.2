import numpy as np
import PIL.Image as pil


def padding(img, padding_size, valor=0):  # caso queira uma borda com valor diferente
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

    padding_size = kernel_shape[0]-2
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


def saveImgFromArray(imgArray, imgName):
    img = pil.fromarray(imgArray)
    img = img.convert('L')
    img.save(imgName)

