# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:43:24 2019

@author: Agostinho
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil 
import cv2
import realce

img_1a = cv2.imread("images/image_1a.png")

r = img_1a[:,:,0]
g = img_1a[:,:,1]
b = img_1a[:,:,2]


cv2.imwrite("r_img1a.png", r)
cv2.imwrite("g_img1a.png", g)
cv2.imwrite("b_img1a.png", b)

#########################################################
                    #Clarear Image_1a
#########################################################
#Linear

#1


newr = realce.linear(r, 1.9, 0, "r_img1a").reshape(400,400,1)
newg = realce.linear(g, 1.3, 0, "g_img1a").reshape(400,400,1)
newb = realce.linear(b, 1.1, 0, "b_img1a").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_linear1.png", newImg1a)

#2

newr = realce.linear(r, 1.95, 0, "r_img1a2").reshape(400,400,1)
newg = realce.linear(g, 1.55, 0, "g_img1a2").reshape(400,400,1)
newb = realce.linear(b, 1.25, 0, "b_img1a2").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_linear2.png", newImg1a)

#Gama

#1

newr = realce.gama(r, 255, 0.5, "r_img1a").reshape(400,400,1)
newg = realce.gama(g, 255, 0.5, "g_img1a").reshape(400,400,1)
newb = realce.gama(b, 255, 0.5, "b_img1a").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_gama1.png", newImg1a)

#2

newr = realce.gama(r, 195, 0.5, "r_img1a").reshape(400,400,1)
newg = realce.gama(g, 195, 0.5, "g_img1a").reshape(400,400,1)
newb = realce.gama(b, 195, 0.5, "b_img1a").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_gama2.png", newImg1a)

#Histograma

#1


newr = realce.histEq(r, 170, "r_img1a").reshape(400,400,1)
newg = realce.histEq(g, 170, "g_img1a").reshape(400,400,1)
newb = realce.histEq(b, 170, "b_img1a").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_hist1.png", newImg1a)

#Log

newr = realce.log(r, 5, "r_img1a").reshape(400,400,1)
newg = realce.log(g, 4, "g_img1a").reshape(400,400,1)
newb = realce.log(b, 3, "b_img1a").reshape(400,400,1)

newImg1a = np.concatenate((newr, newg), axis=2)
newImg1a = np.concatenate((newImg1a, newb), axis=2)
cv2.imwrite("new_img1a_log1.png", newImg1a)
