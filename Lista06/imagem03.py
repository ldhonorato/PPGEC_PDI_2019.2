#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:34:20 2019

@author: timevisao-mk2
"""

import morfologia
import numpy as np
import cv2
import matplotlib.pyplot as plt

morfologia.dilatar("images/image_3.png", np.ones((7,7), np.int8), "image3_dilate_kernel7.png")
morfologia.dilatar("images/image_3.png", np.ones((5,5), np.int8), "image3_dilate_kernel5.png")
morfologia.dilatar("images/image_3.png", np.ones((3,3), np.int8), "image3_dilate_kernel3.png")

morfologia.erodir("image3_dilate_kernel3.png", np.zeros((3,3), np.int8), "image3_delate_erosao_kernel3.png")
morfologia.erodir("image3_dilate_kernel3.png", np.zeros((5,5), np.int8), "image3_delate_erosao_kernel5.png")
morfologia.erodir("image3_delate_erosao_kernel3.png", np.zeros((3,3), np.int8), "image3_delate_erosao2_kernel3.png")


pontos = [(120, 80), (70, 300), (300, 80), (250, 250), (430, 140), (360, 380)]
a = morfologia.preencher("image3_delate_erosao2_kernel3.png", pontos, np.ones((7,7), np.int8), "image3_preenchida.png", 20000)

img1 = cv2.imread('image3_delate_erosao_kernel3.png')
img2 = cv2.imread('image3_preenchida.png')

img3 = cv2.imread('image3_delate_erosao2_kernel3.png')

morfologia.opera(img1, img2, morfologia.OR, 'image3_1erosao_final.png')
morfologia.opera(img3, img2, morfologia.OR, 'image3_2erosao_final.png')

#80 x 120
#300 x 70
#80 x 300
#250 x 250
#140 x 430
#380 x 360
