# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:43:44 2019

@author: Agostinho
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil        

def histEq(img, alfa, nome):
#    img = pil.open(path)
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

    
    img2 = pil.fromarray(img)
    img2 = img2.convert("L")
    img2.save(nome + "_hist.png")
    return img
  
    
    
def gama(img, alfa, beta, nome):
#    img = pil.open(path)
    img = np.array(img).astype(np.float32)/255.
    img = alfa*(img**beta)
    img2 = pil.fromarray(img)
    img2 = img2.convert("L")
    img2.save(nome + "_gama.png")

    return img

def log(img, alfa, nome):
#    img = pil.open(path)
    img = alfa*(np.log10((np.array(img)/255.)+1)).astype(np.float32)*255
    img2 = pil.fromarray(img)
    img2 = img2.convert("L")
    img2.save(nome + "_log.png")

    return img


    
def linear(img, alfa, beta, nome):
#    img = pil.open(path)
    img = np.array(img)*alfa + beta

    img2 = pil.fromarray(img)
    img2 = img2.convert("L")
    img2.save(nome + "_linear.png")
    return img


