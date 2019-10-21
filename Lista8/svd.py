#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:29:43 2019

@author: leandro
"""
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

if __name__ == "__main__":
    paths = ['images/image_1.tif', 'images/image_2.tif', 'images/image_3.tif']
    
    for path in paths:
        prefix = path.split('/')[1][:-4]
        img_original = cv2.imread(path,0)
        tamanho_original = img_original.shape[0]*img_original.shape[1]
        U, sigma, V = np.linalg.svd(img_original)
        print('##########################################')
        print('##########################################')
        print(path)
        print(img_original.shape)
        print('Tamanho Original = ', tamanho_original)
        for i in range(25, 150, 10):
            new_u = U[:, :i]
            new_sigma = sigma[:i]
            new_v = V[:i, :]
            imagem_reconstruida = np.matrix(new_u) * np.diag(new_sigma) * np.matrix(new_v)
            
            tamanho_reduzido = new_u.shape[0]*new_u.shape[1] + new_v.shape[0]*new_v.shape[1] + new_sigma.shape[0]            
            
            print("Tamanho (", str(i), "): ", tamanho_reduzido)
            
            print('---------------------------------')
            taxa_compressao = tamanho_original/tamanho_reduzido
            print('Taxa de compressao (', str(i),") = ", taxa_compressao)
            print('Redundância relativa (', str(i), ') = ', 1-(1/taxa_compressao))
            pil.fromarray(imagem_reconstruida).convert("L").save(prefix + str(i) +'_reconstruida.tif')
