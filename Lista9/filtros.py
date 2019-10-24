#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:04:23 2019

@author: timevisao-mk1
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:55:34 2019

@author: Agostinho
"""
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

def padding(img, camadas=1, valor=255): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+2*camadas, img.shape[1]+2*camadas)) + valor
    limx = newImg.shape[0]
    limy = newImg.shape[1]
    newImg[camadas:limx-camadas, camadas:limy-camadas] = img
    
    return newImg, img.shape


def gauss(subImg):
        
    g = np.array([[1, 4, 7, 4, 1],
                 [4, 16, 26, 16, 4],
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1, 4, 7, 4, 1]])/273
    
    pixel = np.sum(subImg*g)
    
    return pixel

def fGauss(imagem):
    img, origShape = padding(imagem, 2, 255)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - 4):
        for j in range(img.shape[1] - 4):
            ni = i + 2
            nj = j + 2
            newImg[i, j] = gauss(img[ni-2:ni+3,nj-2:nj+3]) #sub matriz
    
    return newImg

def fLaplaciano(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
    filtro = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
      
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = np.sum((img[ni-1:ni+2,nj-1:nj+2])*filtro) #sub matriz
    
    return newImg

def filtro(path, function, repeticoes, nome): #
    img = pil.open(path)
    img = np.array(img)
    
    for rep in range(repeticoes):
        img = function(img)
    img2 = img.copy()
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + ".png")
    
    return img2


