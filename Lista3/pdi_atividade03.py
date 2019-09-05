# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:31:48 2019

@author: Agostinho
"""
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

def saveImgFromArray(imgArray, imgName):
    img = pil.fromarray(imgArray)
    img = img.convert('L')
    img.save(imgName)

def padding(img, valor=0): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]*2, img.shape[1]*2)) + valor
    limx = newImg.shape[0]
    limy = newImg.shape[1]
    newImg[int(limx/4):int(3*limx/4), int(limy/4):int(3*limy/4)] = img
    
    return newImg

def removePadding(img):
    limx = img.shape[0]
    limy = img.shape[1]
    return img[int(limx/4):int(3*limx/4), int(limy/4):int(3*limy/4)]

def padding2(img, valor=0): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]*2, img.shape[1]*2)) + valor
    newImg[0:img.shape[0], 0:img.shape[1]] = img

    return newImg

def removePadding2(img, originalShape):
    return img[0:originalShape[0], 0:originalShape[1]]


def LPF(shape, valor, n=0):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            if dist > valor:
                filtro[i, j] = 0
            else:
                filtro[i, j] = 1
    return filtro
    
def GLPF(shape, valor, n=0):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            filtro[i, j] = np.exp((-(dist)**2)/(2*valor**2))
    return filtro

def BLPF(shape, valor, n=2):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            denominador = 1 + (dist/valor)**(2*n)
            filtro[i, j] = 1/denominador
    return filtro

def BHPF(shape, valor, n=2):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            if dist != 0:
                denominador = 1 + (valor/dist)**(2*n)
                filtro[i, j] = 1/denominador
            else:
                filtro[i, j] = 0
            
    return filtro

def GHPF(shape, valor, n=0):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            filtro[i, j] = 1 -  np.exp((-(dist)**2)/(2*valor**2))
            
                
    return filtro
    
def HPF(shape, valor, n=0):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            if dist < valor:
                filtro[i, j] = 0
            else:
                filtro[i, j] = 1
                
    return filtro

def LAPLA(shape, valor, n=0):
    filtro = np.zeros(shape)
    
    for i in range(shape[0]):        
        for j in range(shape[1]):
            dist = ((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**0.5
            filtro[i, j] = -1*(4*(np.pi**2)*(dist**2))
    
    filtro /= np.max(np.abs(filtro))
    return filtro
    
def aplica(path, valor, tipoFiltro, nome, n=2):
    imageName = path.split('/')[-1]
    imageName = imageName[0:-4]

    img = np.array(pil.open(path))
    originalShape = img.shape
    img = padding2(img)
    img = img/255.
    
    filtro = tipoFiltro(img.shape, valor, n)
                
    imgfft = np.fft.fft2(img)
    imgfftshift = np.fft.fftshift(imgfft)
    
    spectro = 20*np.log(np.abs(imgfftshift))
    
    conv = imgfftshift * filtro
    
    imgIshift = np.fft.ifftshift(conv)
    newImg = np.fft.ifft2(imgIshift)
    newImg = np.abs(newImg)
    
   
    newImg = removePadding2(newImg, originalShape)
    
    #imgfft2 = np.fft.fft2(newImg)
    #imgfftshift2 = np.fft.fftshift(imgfft2)
    
    spectro2 = 20*np.log(np.abs(conv))
    
    im = pil.fromarray(newImg*255.)
    im = im.convert("L")
    im.save(imageName + "_" + nome + ".png")
    
    im2 = pil.fromarray(spectro)
    im2 = im2.convert("L")
    im2.save(imageName + "_spectro.png")
    
    im3 = pil.fromarray(filtro*255.)
    im3 = im3.convert("L")
    im3.save("filtro_" + nome + ".png")
    
    im4 = pil.fromarray(spectro2)
    im4 = im4.convert("L")
    im4.save(imageName + "_" + nome + "_spectro.png")

    return newImg

def filtroAltaEnfaseFrequencia(path, valor, tipoFiltro, nome, k1=1, k2=1, n=2):
    imageName = path.split('/')[-1]
    imageName = imageName[0:-4]

    img = np.array(pil.open(path))
    originalShape = img.shape
    img = padding2(img)
    img = img/255.
    
    filtro = tipoFiltro(img.shape, valor, n)
                
    imgfft = np.fft.fft2(img)
    imgfftshift = np.fft.fftshift(imgfft)
    
    spectro = 20*np.log(np.abs(imgfftshift))
    
    conv = k1 + k2*(imgfftshift * filtro)
    "ENFASE_ALTA_FREQ"
    imgIshift = np.fft.ifftshift(conv)
    newImg = np.fft.ifft2(imgIshift)
    newImg = np.abs(newImg)
  
    newImg = removePadding2(newImg, originalShape)
    
    #imgfft2 = np.fft.fft2(newImg)
    #imgfftshift2 = np.fft.fftshift(imgfft2)
    
    spectro2 = 20*np.log(np.abs(conv))
    
    im = pil.fromarray(newImg*255.)
    im = im.convert("L")
    im.save(imageName + "_" + nome + "AltaEnfase_k1-" + str(k1)+ "_k2-" + str(k2) + ".png")
    
    im2 = pil.fromarray(spectro)
    im2 = im2.convert("L")
    im2.save(imageName + "_spectro.png")
    
    im3 = pil.fromarray(filtro*255.)
    im3 = im3.convert("L")
    im3.save("filtro_" + nome + ".png")
    
    im4 = pil.fromarray(spectro2)
    im4 = im4.convert("L")
    im4.save(imageName + "_" + nome + "_spectro" + "AltaEnfase_k1-" + str(k1)+ "_k2-" + str(k2) + ".png")

def normalizar(img):
    return (img/np.max(img))

def realceHistEq(path, alfa, nome):
    img = pil.open(path)
    img = np.array(img)
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
    
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_hist.png")


