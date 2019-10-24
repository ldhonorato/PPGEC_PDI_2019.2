# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:48:38 2019

@author: Agostinho
"""


import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil 
import cv2

def padding(img, camadas=1, valor=255): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+2*camadas, img.shape[1]+2*camadas)) + valor
    limx = newImg.shape[0]
    limy = newImg.shape[1]
    newImg[camadas:limx-camadas, camadas:limy-camadas] = img
    
    return newImg, img.shape

def mediana(subImg):
    filtro = subImg.flatten()
    filtro.sort()
    return filtro[int(len(filtro)/2)]


def fMediana(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
      
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = mediana(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg

img_2b = cv2.imread("images/image_2b.png")
r = img_2b[:,:,0]
g = img_2b[:,:,1]
b = img_2b[:,:,2]

cv2.imwrite("r_img2b.png", r)
cv2.imwrite("g_img2b.png", g)
cv2.imwrite("b_img2b.png", b)

num_rep_r = 3
num_rep_g = 6
num_rep_b = 4

for rep in range(num_rep_r):
    r = fMediana(r)
for rep in range(num_rep_g):
    g = fMediana(g)
for rep in range(num_rep_b):
    b = fMediana(b)
    
cv2.imwrite("r_img2b_mediana" +str(num_rep_r)+".png", r)
cv2.imwrite("g_img2b_mediana" +str(num_rep_g)+".png", g)
cv2.imwrite("b_img2b_mediana" +str(num_rep_b)+".png", b)

newImg2b = np.concatenate((r.reshape((500,400,1)), g.reshape((500,400,1))), axis=2)
newImg2b = np.concatenate((newImg2b, b.reshape((500,400,1))), axis=2)
cv2.imwrite("new_img2b_mediana4.png", newImg2b)