# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:31:15 2019

@author: Agostinho
"""

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import random
import math
from scipy import ndimage


def padding(img, camadas=1, valor=255): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+2*camadas, img.shape[1]+2*camadas)) + valor
    limx = newImg.shape[0]
    limy = newImg.shape[1]
    newImg[camadas:limx-camadas, camadas:limy-camadas] = img
    
    return newImg, img.shape

def sobel(subImg):
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    resultado = ((np.sum(gx * subImg))**2 + (np.sum(gy * subImg))**2)**0.5
    
    return resultado


def fSobel(imagem):
    img, origShape = padding(imagem, valor=0)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = sobel(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg

def sobel2(subImg):
    gx = np.array([[0, 1, 2],
                   [-1, 0, 1],
                   [-2, -1, 1]])
    
    gy = np.array([[-2, -1, 0],
                   [-1, 0, 1],
                   [0, 1, 2]])
    
    resultado = ((np.sum(gx * subImg))**2 + (np.sum(gy * subImg))**2)**0.5
    
    return resultado


def fSobel2(imagem):
    img, origShape = padding(imagem, valor=0)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = sobel2(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg

def fMedia(imagem):
    img, origShape = padding(imagem, valor =100)
    newImg = np.zeros(origShape)
      
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = math.ceil((np.mean(img[ni-1:ni+2,nj-1:nj+2]))) #sub matriz
    
    return newImg

def marrHildreth(subImg, desvio, dim):
    mask = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            mask[i, j] = ((i**2 + j**2 - desvio**2)/(desvio**4))*np.exp(-((i**2 + j**2)/(2*desvio**2)))
    
    resultado = np.sum(subImg*mask)       
    return resultado, mask

def fMH(imagem, desvio, dim):
    valor = int((dim - 1)/2)    
    img, origShape = padding(imagem, valor, 0)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - valor*2):
        for j in range(img.shape[1] - valor*2):
            ni = i + valor
            nj = j + valor
            newImg[i, j], filtro = marrHildreth(img[ni-valor:ni+valor+1,nj-valor:nj+valor+1], desvio, dim) #sub matriz
    
    return newImg, filtro

def fMediana(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
      
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = np.median(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg


def filtro(path, function, repeticoes, nome, dim=None, desvio=None):
    img = pil.open(path)
    img = np.array(img)
    filtro = None
    for rep in range(repeticoes):
        if desvio != None:
            img, filtro = function(img, desvio, dim)
        elif dim != None:
            img, filtro = function(img, dim)
        else:
            img = function(img)
     
    pil.fromarray(img).convert("L").save(nome)
    return img, filtro

def canny(path, dim, desvio, nome):
    img = pil.open(path)
    img = np.array(img)
    
    valor = int((dim - 1)/2)    
    img, origShape = padding(img, valor, 100)
    newImg = np.zeros(origShape)
    
    gaussian = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            gaussian[i, j] = np.exp(-((i**2 + j**2)/(2*desvio**2)))
            
    for i  in range(img.shape[0] - valor*2):
        for j in range(img.shape[1] - valor*2):
            ni = i + valor
            nj = j + valor
            newImg[i, j] = np.sum(img[ni-valor:ni+valor+1,nj-valor:nj+valor+1]*(gaussian/np.sum(gaussian)))
            
            
    gx = np.array([[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]])

    gy = np.array([[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]])
    
    cx = np.convolve(newImg.flatten(), gx.flatten(), "same").reshape(origShape)
    
    cy = np.convolve(newImg.flatten(), gy.flatten(), "same").reshape(origShape)
    
    M = (cx**2 + cy**2)**0.5
    M = (M/np.max(M))*255
    
    alfa = np.arctan2(cy, cx)
    
    newImg = fSobel(newImg)
    
    newImg = non_max_suppression(newImg, alfa)
    
    pil.fromarray(newImg).convert("L").save(nome)     
    return newImg
    
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z  

dx = [-1,0,1,1,1,0,-1,-1]
dy = [1,1,1,0,-1,-1,-1,0]
def dfs(i, j, c, w, g):
    w[i, j] = c
    for k in range(8):
        ni = i+dy[k]
        nj = j+dx[k]
        if (g[ni, nj] != 0) and (w[ni, nj] == 0): 
            dfs(ni,nj,c, w, g)
    

def findConnectedPoints2(imgBin):
    g, _ = padding(imgBin, valor=0)
    w = np.zeros(g.shape)
    id = 10
    for i in range(1, g.shape[0]-1):
        for j in range(1, g.shape[1]-1):
            if (g[i, j] != 0) and (w[i, j] == 0):
                dfs(i,j,id, w, g)
                id += 1
    return w[1:w.shape[0]-1, 1:w.shape[1]-1]

def findConnectedPoints(imgBin):
    imgBinPad, _ = padding(imgBin, valor=0)
    #img = np.zeros(imgBin.shape)
    id = 10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ni = i + 2
            nj = j + 2
            if imgBinPad[ni, nj] == 1:
                imgBinPad[ni, nj] = id

                offset_dir = 1
                direita = imgBinPad[ni, nj+offset_dir]
                while direita == 1 and (nj+offset_dir < imgBinPad.shape[1]):
                    imgBinPad[ni, nj+offset_dir] = id
                    
                    offset_baixo = 1
                    embaixo = imgBinPad[ni+offset_baixo, nj+offset_dir]
                    while embaixo == 1 and (ni+offset_baixo < imgBinPad.shape[0]):
                        imgBinPad[ni+offset_baixo, nj+offset_dir] = id
                        
                        offset_esquerda = 1
                        esquerda = imgBinPad[ni+offset_baixo, nj+offset_dir-offset_esquerda]
                        while esquerda == 1 and (nj+offset_dir-offset_esquerda > 0):
                            imgBinPad[ni+offset_baixo, nj+offset_dir-offset_esquerda] = id
                            offset_esquerda += 1
                            esquerda = imgBinPad[ni+offset_baixo, nj+offset_dir-offset_esquerda]
                        
                        offset_dir_2 = offset_dir + 1
                        direita2 = imgBinPad[ni+offset_baixo, nj+offset_dir_2]
                        while direita2 == 1 and (nj+offset_dir_2 < imgBinPad.shape[1]):
                            imgBinPad[ni+offset_baixo, nj+offset_dir_2] = id

                            offset_subir = 1
                            emcima = imgBinPad[ni+offset_baixo-offset_subir, nj+offset_dir_2]
                            while emcima == 1 and (ni+offset_baixo-offset_subir > 0):
                                imgBinPad[ni+offset_baixo-offset_subir, nj+offset_dir_2] = id
                                offset_subir += 1
                                emcima = imgBinPad[ni+offset_baixo-offset_subir, nj+offset_dir_2]

                            offset_dir_2 += 1
                            direita2 = imgBinPad[ni+offset_baixo, nj+offset_dir_2]
                        
                        offset_baixo += 1
                        embaixo = imgBinPad[ni+offset_baixo, nj+offset_dir]
                    
                    offset_dir += 1
                    direita = imgBinPad[ni, nj+offset_dir]

                id += 1
    return imgBinPad[1:imgBinPad.shape[0]-1, 1:imgBinPad.shape[1]-1]

    

if __name__ == "__main__":
    """
    a = filtro("images/image_1.png", fMH, 1, "image01_de4_dim5_fMH.png", dim=5, desvio=4)
    limiar_list = [-25,-35,-45,-55,-65,-75,-85,-95]
    for limiar in limiar_list:
        img = a[0] > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_de4_dim5_fMH_limiar_"+str(limiar)+".png")
      
    b = filtro("images/image_1.png", fSobel, 1, "image01_sobel.png")[0]
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = b > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_sobel_limiar_"+str(limiar)+".png")
      
    c = filtro("images/image_1.png", fMediana, 1, "image01_mediana.png")[0]
    d = filtro("image01_mediana.png", fSobel, 1, "image01_media_sobel.png")[0]
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = d > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_mediana_sobel_limiar_"+str(limiar)+".png")
    
    e = filtro("images/image_1.png", fSobel2, 1, "image01_sobel2.png")[0]
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = e > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_sobel2_limiar_"+str(limiar)+".png")
    
    f = canny("images/image_1.png", 5, 0.5, "image01_sobel_canny.png")
    
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = f > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_sobel_canny_limiar_"+str(limiar)+".png")
        
    f = canny("images/image_1.png", 5, 0.2, "image01_sobel_canny.png")
    
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = f > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image01_sobel_canny2_limiar_"+str(limiar)+".png")
    
    g = canny("images/image_2.png", 5, 0.2, "image02_sobel_canny.png")
    
    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        img = g > limiar
        img = img.astype(np.int8)*255
        pil.fromarray(img).convert("L").save("image02_sobel_canny_limiar_"+str(limiar)+".png")
    """
    img = np.array(pil.open('images/image_6.png'))

    limiar_list = range(0, 250, 25)
    for limiar in limiar_list:
        imgL = img > limiar
        imgL = imgL.astype(np.int8)*255
        pil.fromarray(imgL).convert("L").save("Aval_limiar06/image6_limiar_"+str(limiar)+".png")
    
    imgBin = np.array(pil.open('Aval_limiar06/image6_limiar_175.png'))
    imgBin = imgBin/255
    plt.imshow(imgBin, cmap='gray')
    plt.show()
    labeled, nr_objects = ndimage.label(imgBin)
    #labeled = findConnectedPoints2(imgBin)
    #labeled = findConnectedPoints(imgBin)
    print("Number of objects is {}".format(nr_objects))

    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if labeled[i, j] > 0:
                labeled[i, j] += 100
    

    plt.imshow(labeled)
    plt.show()
    

    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if labeled[i][j] != 101: #1 eh a caveira
                labeled[i][j] = 0
    
    imgSeg = img * labeled
    
    pil.fromarray(imgSeg).convert("L").save("image6_segmentada.png")
