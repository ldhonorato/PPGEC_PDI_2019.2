import filtros_espaciais as filtros
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os

path_img_agucar02 = 'images/agucar_02.png'


def main():
    imageName = path_img_agucar02.split('/')[-1]
    imageName = imageName[0:-3]
    
    img = pil.open(path_img_agucar02)
    img = np.array(img)

    name_imgBinarizada = imageName + '_binarizada.png'
    if os.path.exists(name_imgBinarizada):
        imgBinarizada = pil.open(name_imgBinarizada)
        imgBinarizada = np.array(name_imgBinarizada)
    else:
        print('Executando Binarizacao')
        imgBinarizada = np.copy(img)
        imgBinarizada[imgBinarizada > 15] = 255
        filtros.saveImgFromArray(imgBinarizada, name_imgBinarizada)
            
    name_imgMedia = imageName + '_media_3x3.png'
    if os.path.exists(name_imgMedia):
        mean_img = pil.open(name_imgMedia)
        mean_img = np.array(mean_img)
    else:
        print('Executando Filtro Média')
        mean_img = filtros.statistic_filter(imgBinarizada, 3, np.mean)
        filtros.saveImgFromArray(mean_img, name_imgMedia)

    name_imgMedian = imageName + '_gaussian_3x3.png'
    if os.path.exists(name_imgMedian):
        gaussian3x3_img = pil.open(name_imgMedian)
        gaussian3x3_img = np.array(gaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussian3x3_img = filtros.conv(imgBinarizada, filtros.generateGaussianKernel(1.8))
        filtros.saveImgFromArray(gaussian3x3_img, name_imgMedian)

    images = [(img, 'Imagem original'),
              (imgBinarizada, 'Limiar = 15'),
              (mean_img, 'Imagem média 3x3'),
              (gaussian3x3_img, 'Imagem gaussiana')]
    filtros.plotImages(images, 2, 2, 'Resultados ' + imageName)

if __name__ == "__main__":
    main()