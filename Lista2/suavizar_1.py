import filtros_espaciais as filtros
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os

path_img_suavizar01 = 'images/suavizar_01.png'



def main():
    img = pil.open(path_img_suavizar01)
    img = np.array(img)

    if os.path.exists("suavizar_01_media_3x3.png"):
        mean_img = pil.open("suavizar_01_media_3x3.png")
        mean_img = np.array(mean_img)
    else:
        print('Executando Filtro Média')
        mean_img = filtros.statistic_filter(img, 3, np.mean)
        filtros.saveImgFromArray(mean_img, "suavizar_01_media_3x3.png")

    if os.path.exists("suavizar_01_mediana_3x3.png"):
        median_img = pil.open("suavizar_01_mediana_3x3.png")
        median_img=np.array(median_img)
    else:
        print('Executando Filtro Mediana')
        median_img = filtros.statistic_filter(img, 3, np.median)
        filtros.saveImgFromArray(median_img, "suavizar_01_mediana_3x3.png")

    if os.path.exists("suavizar_01_gaussian_3x3.png"):
        gaussian3x3_img = pil.open("suavizar_01_gaussian_3x3.png")
        gaussian3x3_img = np.array(gaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussian3x3_img = filtros.conv(img, filtros.gaussian_kernel_3x3)
        filtros.saveImgFromArray(gaussian3x3_img, "suavizar_01_gaussian_3x3.png")
    
    if os.path.exists("suavizar_01_MedianaGaussiano_3x3.png"):
        medianaGaussian3x3_img = pil.open("suavizar_01_MedianaGaussiano_3x3.png")
        medianaGaussian3x3_img = np.array(medianaGaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        medianaGaussian3x3_img = filtros.conv(median_img, filtros.generateGaussianKernel(1.8))
        filtros.saveImgFromArray(medianaGaussian3x3_img, "suavizar_01_MedianaGaussiano_3x3.png")

    images = [(img, 'Imagem original'),
            (mean_img, 'Filtro média 3x3'),
            (median_img, 'Filtro mediana 3x3'),
            (gaussian3x3_img, 'Filtro gaussiano 3x3'),
            (medianaGaussian3x3_img, 'Filtro Mediana -> gaussiano 3x3')]
    filtros.plotImages(images, 3, 2, 'Resultados Suavizar 01')

    # mean_img = mean_img.convert("L")
    # mean_img.save("suavizar_media3x3_01.png")

if __name__ == "__main__":
    main()