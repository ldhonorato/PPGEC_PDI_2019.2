import filtros_espaciais as filtros
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os

path_img_agucar01 = 'images/agucar_01.png'

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

    name_imgGaussian = imageName + '_gaussian_3x3.png'
    if os.path.exists(name_imgGaussian):
        gaussian3x3_img = pil.open(name_imgGaussian)
        gaussian3x3_img = np.array(gaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussian3x3_img = filtros.conv(img, filtros.gaussian_kernel_3x3)
        filtros.saveImgFromArray(gaussian3x3_img, name_imgGaussian)

    name_imgGaussian18 = imageName + '_gaussianSigma_18_3x3.png'
    if os.path.exists(name_imgGaussian18):
        gaussianSigma3x3_img = pil.open(name_imgGaussian18)
        gaussianSigma3x3_img = np.array(gaussianSigma3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussianSigma3x3_img = filtros.conv(img, filtros.generateGaussianKernel(1.8))
        filtros.saveImgFromArray(gaussianSigma3x3_img, name_imgGaussian18)
    
    name_imgLaplace = imageName + '_laplacian_3x3.png'
    laplace3x3_img = filtros.conv(img, filtros.laplace_kernel_3x3)
    filtros.saveImgFromArray(laplace3x3_img, name_imgLaplace)

    name_imgLaplaceAjustado = imageName + '_laplacianAjustado_3x3.png'
    laplaceAjustado3x3_img = laplace3x3_img - np.min(laplace3x3_img)
    filtros.saveImgFromArray(laplaceAjustado3x3_img, name_imgLaplaceAjustado)

    name_imgLaplaceDiag = imageName + '_laplacian_diagonal_3x3.png'
    laplaceDiagonal3x3_img = filtros.conv(img, filtros.laplace_diagonal_kernel_3x3)
    filtros.saveImgFromArray(laplaceDiagonal3x3_img, name_imgLaplaceDiag)

    name_imgLaplaceDiagAjustado = imageName + '_laplacian_diagonalAjustado_3x3.png'
    laplaceDiagonalAjustado3x3_img = laplaceDiagonal3x3_img - np.min(laplaceDiagonal3x3_img)
    filtros.saveImgFromArray(laplaceDiagonalAjustado3x3_img, name_imgLaplaceDiagAjustado)

    #agucamento Laplaciano
    name_imgAucamentoLaplace = imageName + '_agucamentoLaplace.png'
    agucamentoLaplace = img - laplace3x3_img
    filtros.saveImgFromArray(agucamentoLaplace, name_imgAucamentoLaplace)

    name_imgAucamentoLaplaceDiagonal = imageName + '_agucamentoLaplaceDiagonal.png'
    agucamentoLaplaceDiag = img - laplaceDiagonal3x3_img
    filtros.saveImgFromArray(agucamentoLaplaceDiag, name_imgAucamentoLaplaceDiagonal)

    name_imgAucamentoLaplaceAjustado = imageName + '_agucamentoLaplaceAjustado.png'
    agucamentoLaplaceAjustado = img - laplaceAjustado3x3_img
    filtros.saveImgFromArray(agucamentoLaplaceAjustado, name_imgAucamentoLaplaceAjustado)

    name_imgAucamentoLaplaceDiagonalAjustado = imageName + '_agucamentoLaplaceDiagAjustado.png'
    agucamentoLaplaceDiagAjustado = img - laplaceDiagonalAjustado3x3_img
    filtros.saveImgFromArray(agucamentoLaplaceDiagAjustado, name_imgAucamentoLaplaceDiagonalAjustado)

    #Mascara de Nitidez
    name_MascaraNitidez = imageName + '_MascaraNitidezGaussiana.png'
    mask = img - gaussian3x3_img
    filtros.saveImgFromArray(mask, name_MascaraNitidez)
    
    name_MascaraNitidez = imageName + '_MascaraNitidez.png'
    unsharpMaskResult_img = img + mask
    filtros.saveImgFromArray(unsharpMaskResult_img, name_MascaraNitidez)

    #High Boost
    name_HighBoost_k2 = imageName + '_HighBoost_k2.png'
    hihgBoostResult_img = img + 2*mask
    filtros.saveImgFromArray(hihgBoostResult_img, name_HighBoost_k2)

    name_HighBoost_k4 = imageName + '_HighBoost_k4.png'
    hihgBoostResult4_img = img + 4*mask
    filtros.saveImgFromArray(hihgBoostResult4_img, name_HighBoost_k4)

    plt.figure()
    plt.hist(img.flatten(), 256, [0, 255])
    plt.show()

    images = [(img, 'Imagem original'),
              (mean_img, 'Filtro média 3x3'),
              (median_img, 'Filtro mediana 3x3'),
              (gaussian3x3_img, 'Filtro gaussiano 3x3'),
              (laplace3x3_img, 'Filtro Laplace 3x3'),
              (laplaceDiagonal3x3_img, 'Filtro Laplace Diagonal 3x3'),
              (laplaceAjustado3x3_img, 'Filtro Laplace Ajustado 3x3'),
              (laplaceDiagonalAjustado3x3_img, 'Filtro Laplace Diagonal Ajustado 3x3'),
              (agucamentoLaplace, 'Agucamento Laplace'),
              (agucamentoLaplaceDiag, 'Agucamento Laplace Diagonal'),
              (agucamentoLaplaceAjustado, 'Agucamento Laplace Ajustado'),
              (agucamentoLaplaceDiagAjustado, 'Agucamento Laplace Diagonal Ajustado'),
              (unsharpMaskResult_img, 'Agucamento Mascara Nitidez (gaussiano)'),
              (mask, 'mascara gaussiana'),
              (hihgBoostResult_img, 'Agucamento High Boost (gaussiano k = 2)'),
              (hihgBoostResult4_img, 'Agucamento High Boost (gaussiano k = 4)')]
    filtros.plotImages(images, 4, 4, 'Resultados ' + imageName)
  
if __name__ == "__main__":
    main()