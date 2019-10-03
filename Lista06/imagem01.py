#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:54:34 2019

@author: timevisao-mk2
"""

import morfologia
import numpy as np
#mport PIL.Image as pil
import cv2

path_original = ["images/image_1a.png", "images/image_1b.png", "images/image_1c.png",
                 "images/image_1d.png", "images/image_1e.png", "images/image_1f.png"]

img_originais = []

for path in path_original:
    img_originais.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

#image_1a = 0
#image_1b = 1
#image_1c = 2
#image_1d = 3
#image_1e = 4    
#image_1f = 5

res_a_and_b = morfologia.opera(img_originais[0], img_originais[1], morfologia.AND, "res1_a_and_b.png")
res_d_and_e = morfologia.opera(img_originais[3], img_originais[4], morfologia.AND, "res2_d_and_e.png")

not_c = morfologia.opera(img_originais[2], img_originais[2], morfologia.NAND, "res3_not_c.png")

res_not_c_and_res1 = morfologia.opera(res_a_and_b, not_c, morfologia.AND, "res4_not_c_and_res1.png")

res_res2_xor_f = morfologia.opera(res_d_and_e, img_originais[5], morfologia.XOR, "res5_res2_xor_f.png")

not_res4 = morfologia.opera(res_not_c_and_res1, res_not_c_and_res1, morfologia.NAND, "res6_not_res4.png")

final = morfologia.opera(not_res4, res_res2_xor_f, morfologia.NAND, "res7_not_res4_or_res5.png")