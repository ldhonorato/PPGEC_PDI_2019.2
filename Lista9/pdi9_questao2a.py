#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:32:18 2019

@author: Agostinho
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil 
import cv2
import filtros

def RGB2HSI(img):
    B = img[:,:,0].astype(np.float)
    G = img[:,:,1].astype(np.float)
    R = img[:,:,2].astype(np.float)
    
#    cv2.imwrite("1r.png", R)
#    cv2.imwrite("1g.png", G)
#    cv2.imwrite("1b.png", B)
    
    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)
    
    i = (R+G+B)/(3*255)
    
    s = 1 - 3*np.minimum.reduce([r, g, b])
    h = np.arccos((0.5*((r-g)+(r-b))) / ((((r-g)**2 + (r-b)*(g-b))**0.5)+0.0001))
    h[b>g] = (2*np.pi)-h[b>g]
    
    H = h*180/np.pi
    S = s*100
    I = i*255
    
    return H, S, I

def HSI2RGB(H, S, I):
    h = H * (np.pi/180)
    s = S/100
    i = I/255
    
    x = i*(1-s)
    y = i*(1 + ((s*np.cos(h)) / np.cos(np.pi/3 - h)))
    z = 3*i-(x+y)
    
    r_new = np.zeros((500,400))
    g_new = np.zeros((500,400))
    b_new = np.zeros((500,400))
    
    for index_0 in range(h.shape[0]):
        for index_1 in range(h.shape[1]):
            if h[index_0, index_1] < 2*np.pi/3:
                b_new[index_0, index_1] = x[index_0, index_1]
                r_new[index_0, index_1] = y[index_0, index_1]
                g_new[index_0, index_1] = z[index_0, index_1]
            elif h[index_0, index_1] < 4*np.pi/3:
                r_new[index_0, index_1] = x[index_0, index_1]
                g_new[index_0, index_1] = y[index_0, index_1]
                b_new[index_0, index_1] = z[index_0, index_1]
            else:
                g_new[index_0, index_1] = x[index_0, index_1]
                b_new[index_0, index_1] = y[index_0, index_1]
                r_new[index_0, index_1] = z[index_0, index_1]
    
    return r_new, g_new, b_new

#######################################################################################
#------------------------TRABALHANDO COM A IMAGEM EM HSI------------------------------
#######################################################################################
#Conversão RGB para HSI
img_2a = cv2.imread("images/image_2a.png")
H, S, I = RGB2HSI(img_2a)
cv2.imwrite("1H.png", ((H/np.max(H))*255))
cv2.imwrite("1S.png", ((S/np.max(S))*255))
cv2.imwrite("1I.png", I)

newI = filtros.filtro("1I.png", filtros.fGauss, 1, "2_newI_umavez")

r, g, b = HSI2RGB(H, S, newI)

r *= 255
g *= 255
b *= 255

cv2.imwrite("3r.png", r)
cv2.imwrite("3g.png", g)
cv2.imwrite("3b.png", b)

img_reconstruida = np.stack((b, g, r), axis=2)
cv2.imwrite("imagem_reconstruida.png", img_reconstruida)


#######################################################################################
#------------------------TRABALHANDO COM A IMAGEM EM RGB------------------------------
#######################################################################################
img_2a = cv2.imread("images/image_2a.png")
new_img2a = np.zeros((500,400,3))

b = img_2a[:,:,0]
g = img_2a[:,:,1]
r = img_2a[:,:,2]

cv2.imwrite("b_img2a.png", b)
cv2.imwrite("g_img2a.png", g)
cv2.imwrite("r_img2a.png", r)

#Componentes Filtradas
newH = filtros.filtro("b_img2a.png", filtros.fGauss, 2, "b_new_img2a_gauss").reshape(500,400,1)
newS = filtros.filtro("g_img2a.png", filtros.fGauss, 2, "g_new_img2a_gauss").reshape(500,400,1)
newI = filtros.filtro("r_img2a.png", filtros.fGauss, 2, "r_new_img2a_gauss").reshape(500,400,1)

#Laplaciano
newH_lapla = filtros.filtro("b_new_img2a_gauss.png", filtros.fLaplaciano, 1, "b_new_img2a_gauss_lapla").reshape(500,400,1)
newS_lapla = filtros.filtro("g_new_img2a_gauss.png", filtros.fLaplaciano, 1, "g_new_img2a_gauss_lapla").reshape(500,400,1)
newI_lapla = filtros.filtro("r_new_img2a_gauss.png", filtros.fLaplaciano, 1, "r_new_img2a_gauss_lapla").reshape(500,400,1)



new_img2a = np.concatenate((newH, newS), axis=2)
new_img2a = np.concatenate((new_img2a, newI), axis=2)
cv2.imwrite("new_img2a_gauss.png", new_img2a)

new_img2a_lapla = np.concatenate((newH_lapla, newS_lapla), axis=2)
new_img2a_lapla = np.concatenate((new_img2a_lapla, newI_lapla), axis=2)
cv2.imwrite("new_img2a_gauss_lapla.png", new_img2a_lapla)

imagem_agucada = new_img2a + new_img2a_lapla
cv2.imwrite("new_img2a_gauss_lapla_agucada.png", imagem_agucada)

