#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:54:34 2019

@author: timevisao-mk2
"""

import morfologia
import numpy as np
import cv2

path_original = "images/image_2.png"




morfologia.dilatar(path_original, np.ones((3,3)), 'image_2_diatar3.png')
morfologia.erodir('image_2_diatar3.png', np.zeros((5,5)), 'image_2_diatar3-erosao5.png')


morfologia.dilatar('image_2_diatar3-erosao5.png', np.ones((3,3)), 'image_2_diatar3-erosao5-dilatar3.png')
morfologia.erodir('image_2_diatar3-erosao5-dilatar3.png', np.zeros((5,5)), 'image_2_diatar3-erosao5-dilatar3-erosao5.png')

morfologia.dilatar('image_2_diatar3-erosao5-dilatar3-erosao5.png', np.ones((3,3)), 'image_2_diatar3-erosao5-dilatar3-erosao5-dilatar3.png')
morfologia.erodir('image_2_diatar3-erosao5-dilatar3-erosao5-dilatar7.png', np.zeros((3,3)), 'image_2_diatar3-erosao5-dilatar3-erosao5-dilatar7-erodir3.png')
