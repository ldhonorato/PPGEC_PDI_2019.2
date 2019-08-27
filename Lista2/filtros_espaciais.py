import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math
import sys

def padding(img, padding_size, valor=0): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+padding_size*2, img.shape[1]+padding_size*2)) + valor
    i = (padding_size-1)
    newImg[i:img.shape[0]+i, i:img.shape[1]+i] = img
    
    return newImg, img.shape

'''
img é um np.array representando uma imagem 2D (MxN)
filter_kernel é um np.array (AxA) (matriz quadrada), onde A é impar (A=2a+1)
'''
def conv(img, filter_kernel):
    kernel_shape = filter_kernel.shape
    assert len(kernel_shape) == 2 #kernel é uma matriz 2D
    assert kernel_shape[0] == kernel_shape[1] #filter kernel é matriz quadrada
    assert kernel_shape[0]%2 == 1 #o kernel tem dimensões impar
    assert kernel_shape[0] > 2 #kernel minimo 3x3

    padding_size = kernel_shape[0]-2
    kernel_sum = np.sum(filter_kernel)
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    for i  in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            newImg[i, j] = np.sum((padded_img[ni-padding_size:ni+(padding_size+1),nj-padding_size:nj+(padding_size+1)])*filter_kernel) #sub matriz
            if kernel_sum != 0:
                newImg[i, j] = newImg[i, j] / kernel_sum

    return newImg

def statistic_filter(img, kernel_size, statistic_function):
    padding_size = kernel_size - 2
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    for i  in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            subImg = padded_img[ni-padding_size:ni+(padding_size+1),nj-padding_size:nj+(padding_size+1)]
            newImg[i, j] = statistic_function(subImg)

    return newImg

path_img_agucar01 = 'images/suavizar_01.png'

gaussian_kernel_5x5 = [[1, 4,  6,  4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4,  6,  4, 1]]

gaussian_kernel_3x3 = [[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]

def main():
    img = pil.open(path_img_agucar01)
    img = np.array(img)
    
    print('Executando Filtro Média')
    mean_img = statistic_filter(img, 3, np.mean)
    
    print('Executando Filtro Mediana')
    median_img = statistic_filter(img, 3, np.median)
    
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', label='Imagem Original')
    fig.add_subplot(1, 3, 2)
    plt.imshow(mean_img, cmap='gray', label='Filtro média 3x3')
    fig.add_subplot(1, 3, 3)
    plt.imshow(median_img, cmap='gray', label='Filtro mediana 3x3')
    plt.show()
    
    #mean_img = mean_img.convert("L")
    #mean_img.save("suavizar_media3x3_01.png")

if __name__ == "__main__":
    main()
