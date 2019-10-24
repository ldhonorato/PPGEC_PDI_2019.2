# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:00:38 2019

@author: Agostinho
"""


import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil 
import cv2
import realce

img_1b = cv2.imread("images/image_1b.png")

r = img_1b[:,:,0]
g = img_1b[:,:,1]
b = img_1b[:,:,2]


cv2.imwrite("r_img1b.png", r)
cv2.imwrite("g_img1b.png", g)
cv2.imwrite("b_img1b.png", b)

#########################################################
                    #Escurer Image_1b
#########################################################
#Linear

#1


newr = realce.linear(r, 0.7, -15, "r_img1b").reshape(400,400,1)
newg = realce.linear(g, 0.5, -15, "g_img1b").reshape(400,400,1)
newb = realce.linear(b, 0.6, -15, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_linear1.png", newimg1b)

#2
newr = realce.linear(r, 0.9, -70, "r_img1b").reshape(400,400,1)
newg = realce.linear(g, 0.6, -50, "g_img1b").reshape(400,400,1)
newb = realce.linear(b, 0.8, -60, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_linear2.png", newimg1b)

#Gama

#1

newr = realce.gama(r, 160, 2, "r_img1b").reshape(400,400,1)
newg = realce.gama(g, 160, 2.4, "g_img1b").reshape(400,400,1)
newb = realce.gama(b, 160, 2, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_gama1.png", newimg1b)

#2

newr = realce.gama(r, 180, 2.1, "r_img1b").reshape(400,400,1)
newg = realce.gama(g, 180, 2.4, "g_img1b").reshape(400,400,1)
newb = realce.gama(b, 180, 2.2, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_gama2.png", newimg1b)

#Histograma

#1


newr = realce.histEq(r, 100, "r_img1b").reshape(400,400,1)
newg = realce.histEq(g, 100, "g_img1b").reshape(400,400,1)
newb = realce.histEq(b, 100, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_hist1.png", newimg1b)

#Log

newr = realce.log(r, 0.8, "r_img1b").reshape(400,400,1)
newg = realce.log(g, 0.7, "g_img1b").reshape(400,400,1)
newb = realce.log(b, 0.9, "b_img1b").reshape(400,400,1)

newimg1b = np.concatenate((newr, newg), axis=2)
newimg1b = np.concatenate((newimg1b, newb), axis=2)
cv2.imwrite("new_img1b_log1.png", newimg1b)
