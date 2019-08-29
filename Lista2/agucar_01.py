import filtros_espaciais as filtros
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os

path_img_agucar01 = 'images/agucar_01.png'

def realceHistEq(img, alfa, nome):
    w = img.shape[0]
    h = img.shape[1]
    total = w*h
    
    cont = {}
    probCom = {}
        
    for pixel in img.flatten():
        if pixel not in cont.keys():
            cont[pixel] = 1
        else:
            cont[pixel] += 1
    
    for entry in cont.keys():
        cont[entry] /= total
    
    itens = list(cont)
    itens.sort()
    temp = 1
    for x in itens:
        v = 0
        for y in itens[:temp]:
            v += cont[y]
        probCom[x] = int(v*alfa)
        temp += 1
    
    for i in range(w):
        for j in range(h):
            img[i, j] = probCom[img[i, j]]
    
    return img

def main():
    imageName = path_img_agucar01.split('/')[-1]
    imageName = imageName[0:-3]
    
    img = pil.open(path_img_agucar01)
    img = np.array(img)

    name_imgMedia = imageName + '_media_3x3.png'
    if os.path.exists(name_imgMedia):
        mean_img = pil.open(name_imgMedia)
        mean_img = np.array(mean_img)
    else:
        print('Executando Filtro Média')
        mean_img = filtros.statistic_filter(img, 3, np.mean)
        filtros.saveImgFromArray(mean_img, name_imgMedia)

    name_imgMedian = imageName + '_mediana_3x3.png'
    if os.path.exists(name_imgMedian):
        median_img = pil.open(name_imgMedian)
        median_img=np.array(median_img)
    else:
        print('Executando Filtro Mediana')
        median_img = filtros.statistic_filter(img, 3, np.median)
        filtros.saveImgFromArray(median_img, name_imgMedian)

    name_imgMedian = imageName + '_gaussian_3x3.png'
    if os.path.exists(name_imgMedian):
        gaussian3x3_img = pil.open(name_imgMedian)
        gaussian3x3_img = np.array(gaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussian3x3_img = filtros.conv(img, filtros.gaussian_kernel_3x3)
        filtros.saveImgFromArray(gaussian3x3_img, name_imgMedian)

    name_imgLaplace = imageName + '_laplacian_3x3.png'
    if os.path.exists(name_imgLaplace):
        laplace3x3_img = pil.open(name_imgLaplace)
        laplace3x3_img = np.array(laplace3x3_img)
    else:
        print('Executando Filtro Laplace')
        laplace3x3_img = filtros.conv(img, filtros.laplace_kernel_3x3)
        filtros.saveImgFromArray(laplace3x3_img, name_imgLaplace)

    laplaceAjustado3x3_img = laplace3x3_img - np.min(laplace3x3_img)

    name_imgLaplaceDiag = imageName + '_laplacian_diagonal_3x3.png'
    if os.path.exists(name_imgLaplaceDiag):
        laplaceDiagonal3x3_img = pil.open(name_imgLaplaceDiag)
        laplaceDiagonal3x3_img = np.array(laplaceDiagonal3x3_img)
    else:
        print('Executando Filtro Laplace Diagonal')
        laplaceDiagonal3x3_img = filtros.conv(img, filtros.laplace_diagonal_kernel_3x3)
        filtros.saveImgFromArray(laplaceDiagonal3x3_img, name_imgLaplaceDiag)

    laplaceDiagonalAjustado3x3_img = laplaceDiagonal3x3_img - np.min(laplaceDiagonal3x3_img)

    gaussianSigma3x3_img = filtros.conv(img, filtros.generateGaussianKernel(1.8))

    #agucamento Laplaciano
    agucamentoLaplace = img - laplace3x3_img
    agucamentoLaplaceDiag = img - laplaceDiagonal3x3_img

    agucamentoLaplaceAjustado = img - laplaceAjustado3x3_img
    agucamentoLaplaceDiagAjustado = img - laplaceDiagonalAjustado3x3_img

    #Mascara de Nitidez
    mask = img - gaussian3x3_img
    unsharpMaskResult_img = img + mask

    mask2 = img - gaussianSigma3x3_img

    hihgBoostResult2_img = img + 2*mask2
    

    #High Boost
    hihgBoostResult_img = img + 2*mask

    hihgBoostResult4_img = img + 4*mask

    plt.figure()
    plt.hist(img.flatten(), 256, [0, 255])
    plt.show()

    images = [(img, 'Imagem original'),
              #(mean_img, 'Filtro média 3x3'),
              #(median_img, 'Filtro mediana 3x3'),
              #(gaussian3x3_img, 'Filtro gaussiano 3x3'),
              #(laplace3x3_img, 'Filtro Laplace 3x3'),
              #(laplaceDiagonal3x3_img, 'Filtro Laplace Diagonal 3x3'),
              #(laplaceAjustado3x3_img, 'Filtro Laplace Ajustado 3x3'),
              #(laplaceDiagonalAjustado3x3_img, 'Filtro Laplace Diagonal Ajustado 3x3'),
              #(agucamentoLaplace, 'Agucamento Laplace'),
              #(agucamentoLaplaceDiag, 'Agucamento Laplace Diagonal'),
              #(agucamentoLaplaceAjustado, 'Agucamento Laplace Ajustado'),
              #(agucamentoLaplaceDiagAjustado, 'Agucamento Laplace Diagonal Ajustado'),
              #(unsharpMaskResult_img, 'Agucamento Mascara Nitidez (gaussiano)'),
              (mask, 'mascara gaussiana'),
              (mask2, 'mascara gaussiana sigma 1.8'),
              (hihgBoostResult2_img, 'High boost (gaussiano sigma=0.85 k = 2'),
              (hihgBoostResult_img, 'Agucamento High Boost (gaussiano k = 2)'),
              (hihgBoostResult4_img, 'Agucamento High Boost (gaussiano k = 4)')]
    filtros.plotImages(images, 3, 3, 'Resultados ' + imageName)
  
if __name__ == "__main__":
    main()