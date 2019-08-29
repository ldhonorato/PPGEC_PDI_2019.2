import filtros_espaciais as filtros
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import os

path_img_suavizar01 = 'images/suavizar_02.png'

def main():
    img = pil.open(path_img_suavizar01)
    img = np.array(img)

    if os.path.exists("suavizar_02_media_3x3.png"):
        mean_img = pil.open("suavizar_02_media_3x3.png")
        mean_img = np.array(mean_img)
    else:
        print('Executando Filtro Média')
        mean_img = filtros.statistic_filter(img, 3, np.mean)
        filtros.saveImgFromArray(mean_img, "suavizar_02_media_3x3.png")

    if os.path.exists("suavizar_02_mediana_3x3.png"):
        median_img = pil.open("suavizar_02_mediana_3x3.png")
        median_img=np.array(median_img)
    else:
        print('Executando Filtro Mediana')
        median_img = filtros.statistic_filter(img, 3, np.median)
        filtros.saveImgFromArray(median_img, "suavizar_02_mediana_3x3.png")

    if os.path.exists("suavizar_02_gaussian_3x3.png"):
        gaussian3x3_img = pil.open("suavizar_02_gaussian_3x3.png")
        gaussian3x3_img = np.array(gaussian3x3_img)
    else:
        print('Executando Filtro Gaussiano')
        gaussian3x3_img = filtros.conv(img, filtros.gaussian_kernel_3x3)
        filtros.saveImgFromArray(gaussian3x3_img, "suavizar_02_gaussian_3x3.png")

    images = [(img, 'Imagem original'),
              (mean_img, 'Filtro média 3x3'),
              (median_img, 'Filtro mediana 3x3'),
              (gaussian3x3_img, 'Filtro gaussianp 3x3')]
    filtros.plotImages(images, 2, 2, 'Resultados Suavizar 02')
  
if __name__ == "__main__":
    main()