#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:54:34 2019

@author: timevisao-mk2
"""

import morfologia
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as pil


path_original = "images/image_4.png"


img_original = cv2.imread(path_original, cv2.IMREAD_GRAYSCALE)
img_bin = (img_original > 100)*1

morfologia.saveImg(img_bin, 'image4_bin.png')
img_bin_e3 = morfologia.erodir('image4_bin.png', np.zeros((3,3)), 'image4_bin_e3.png')



image4_bin_preenchimento1 = morfologia.preencher('image4_bin.png', [(125,93),(137,100),(142, 99)], np.ones((3,3)), 'image4_bin_preenchimento1.png', 3)

morfologia.saveImg(image4_bin_preenchimento1, 'image4_bin_preenchimento1.png')

image4_bin_preenchimento2 = morfologia.preencher('image4_bin.png', [(46,219), (187,108),(80,166),(235,364),(261, 350), (393,225)], np.ones((3,3)), 'image4_bin_preenchimento2.png', 5)

morfologia.saveImg(image4_bin_preenchimento2, 'image4_bin_preenchimento2.png')


img_preenchimento_1 = cv2.imread("image4_bin_preenchimento.png", cv2.IMREAD_GRAYSCALE)
img_preenchimento_2 = cv2.imread("image4_bin_preenchimento1.png", cv2.IMREAD_GRAYSCALE)
img_preenchimento_3 = cv2.imread("image4_bin_preenchimento2.png", cv2.IMREAD_GRAYSCALE)

imagem_final = img_bin*255. + img_preenchimento_1 + img_preenchimento_2 + img_preenchimento_3

pil.fromarray(imagem_final).convert("L").save("image4_bin_preenchida_final.png")
image4_bin_preenchida_final_e3 = morfologia.erodir('image4_bin_preenchida_final.png', np.zeros((5,5)), 'image4_bin_preenchida_final_e5.png')


#img_fronteira = img_bin - img_bin_e3
#morfologia.saveImg(img_fronteira, 'image4_bin_e3_fronteira.png')
#
#img_bin_not = np.logical_not(img_bin)*1
#morfologia.saveImg(img_bin_not, 'img_bin_not.png')
#
#img_bin_d3 = cv2.imread('image4_bin_d3.png', cv2.IMREAD_GRAYSCALE)
#morfologia.opera(img_bin_not, img_bin_d3, morfologia.AND, 'img4_preenchimento.png')

#kernel = np.array([[0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0],
#                   [1, 1, 1, 1, 1],
#                   [0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0]], np.uint8)
#
#
#img_bin_e3_e3 = cv2.erode(cv2.imread('image4_bin_e3.png', cv2.IMREAD_GRAYSCALE), kernel, iterations=1)#morfologia.erodir('image4_bin_e3.png', kernel, 'image4_bin_e3_e5_hor.png')
#morfologia.saveImg(img_bin_e3_e3/255, 'image4_bin_e3_erode_opencv2.png')

#img_bin_e3_e3 = morfologia.erodir('image4_bin_e3.png', np.zeros((3,3)), 'image4_bin_e3_e3.png')
#img_fronteira_2 = img_bin_e3 - img_bin_e3_e3
#morfologia.saveImg(img_fronteira_2, 'image4_bin_e3_e3_fronteira.png')
#
#
#img_bin_e3_e3_e3 = morfologia.erodir('image4_bin_e3_e3.png', np.zeros((3,3)), 'image4_bin_e3_e3_e3.png')
#img_fronteira_3 = img_bin_e3_e3 - img_bin_e3_e3_e3
#morfologia.saveImg(img_fronteira_3, 'image4_bin_e3_e3_e3_fronteira3.png')


#
#img_bin_d3 = morfologia.dilatar('image4_bin.png', np.ones((3,3)), 'image4_bin_d3.png')
#
#img_dif = img_bin_d3 - img_fronteira
#morfologia.saveImg(img_dif, 'image4_bin_d3-fronteira.png')
#
#morfologia.erodir('image4_bin_d3-fronteira.png', np.zeros((3,3)), 'image4_bin_d3-fronteira-e3.png')
#morfologia.dilatar('image4_bin_d3-fronteira.png', np.ones((3,3)), 'image4_bin_d3-fronteira-d3.png')
#
#img1 = morfologia.erodir('image4_bin_d3-fronteira-d3.png', np.zeros((3,3)), 'image4_bin_d3-fronteira-d3-e3.png')
#
#img2 = img1 - img_bin
#morfologia.saveImg(img2,'image4_bin_d3-fronteira-d3-e3_menos_binaria.png')
#
#morfologia.opera(img2, img_fronteira, morfologia.OR, 'image4_dif_fronteiras.png')
#
#kernel = np.array([[0, 0, 0],
#                   [1, 1, 1],
#                   [0, 0, 0]])
#
#morfologia.dilatar('image4_bin_e3_fronteira.png', kernel, 'image4_bin_e3_fronteira_d3_horizontal.png')
#

#
#thresholdWindowsSize = 33
#threshold = 33*33/2
#img_bin_e3_threshold = morfologia.erodirThreshold('image4_bin.png', np.zeros((3,3)), threshold, thresholdWindowsSize,'image4_bin_e3_threshold.png')
#img_fronteira_threshold = img_bin - img_bin_e3_threshold
#morfologia.saveImg(img_fronteira_threshold, 'image4_bin_e3_fronteira_threshold.png')


#morfologia.dilatar('image4_bin.png', np.ones((3,3)), 'image4_bin_d3.png')
#morfologia.erodir('image4_bin_d3.png', np.zeros((5,5)), 'image4_bin_d3_e5.png')
#
#

#morfologia.erodir('image4_bin_e3.png', np.ones((5,5)), 'image4_bin_e3_d5.png')

