#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:07:57 2019

@author: timevisao-mk2
"""

import numpy as np
import PIL.Image as pil
import cv2

def findBzero(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                b0 = (i, j)
                return b0

def findNextB_C(img, b, c):
    caminho = {(b[0]-1,b[1]-1):(b[0]-1,b[1]),
               (b[0]-1,b[1]):(b[0]-1,b[1]+1),
               (b[0]-1,b[1]+1):(b[0],b[1]+1),
               (b[0],b[1]+1):(b[0]+1,b[1]+1),
               (b[0]+1,b[1]+1):(b[0]+1,b[1]),
               (b[0]+1,b[1]):(b[0]+1,b[1]-1),
               (b[0]+1,b[1]-1):(b[0],b[1]-1),
               (b[0],b[1]-1):(b[0]-1,b[1]-1)}
    
    coord_atual = c
    for temp in range(8):
        coord = caminho[coord_atual]
        if img[coord[0], coord[1]] == 1:
            return coord, coord_atual
        coord_atual = coord

def fronteira(path, nome):
    img = cv2.imread(path, 0)/255
    
    b0 = findBzero(img)
    c0 = (b0[0], b0[1] - 1)
    
    b = b0 # no inicio sao iguais
    c = c0
    
    coord_list = []
    newImg = np.zeros(img.shape)
        
    while(True):
        coord_list.append(b)
        newImg[b[0], b[1]] = 1
        b, c = findNextB_C(img, b, c)

        if b in coord_list:
            new_b, new_c = findNextB_C(img, b, c)
            if new_b in coord_list:
                break
    
    pil.fromarray(newImg*255).convert("L").save(nome)
    
    return newImg, coord_list

def create_grid(img_shape, steps):
    newImg = np.zeros(img_shape)
    coord_list_grid = []
    for i in range(0, img_shape[0], steps):
        for j in range(0, img_shape[1], steps):
            coord_list_grid.append((i, j))
            newImg[i, j] = 1
    return newImg, coord_list_grid

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        
def resample(border_img, coord_list, min_dist, grid_steps):
    grid, coord_grid = create_grid(border_img.shape, grid_steps)
    resample_list = []
    newImg = np.zeros(border_img.shape)
    
    for p1 in coord_grid:
        for p2 in coord_list:
            if distance(p1, p2) < min_dist:
                if p1 not in resample_list:
                    resample_list.append(p1)
                    newImg[p1[0], p1[1]] = 1
    
    return newImg, resample_list, grid

def nextPixel(ponto, lista):
    menor = float("inf")
    ponto_prox = None
    for p in lista:
        dist = distance(ponto, p)
        if dist  <= menor:
            menor = dist
            ponto_prox = p
    return ponto_prox

def get8chainDirection(a, b):
    direcoes = {(-1,-1) :"3",
                (-1,0)  :"2",
                (-1,1)  :"1",
                (0,1)   :"0",
                (1,1)   :"7",
                (1,0)   :"6",
                (1,-1)  :"5",
                (0,-1)  :"4"}
    
    d = (b[0]-a[0], b[1]-a[1])
    return direcoes[d]

def get4chainDirection(a, b):
    direcoes = {(-1,-1) :"_",
                (-1,0)  :"1",
                (-1,1)  :"_",
                (0,1)   :"0",
                (1,1)   :"_",
                (1,0)   :"3",
                (1,-1)  :"_",
                (0,-1)  :"2"}
    
    d = (b[0]-a[0], b[1]-a[1])
    return direcoes[d]

def isDiagonal(a, b):
    direcoes = {(-1,-1) :True,
                (-1,0)  :False,
                (-1,1)  :True,
                (0,1)   :False,
                (1,1)   :True,
                (1,0)   :False,
                (1,-1)  :True,
                (0,-1)  :False}
    d = (b[0]-a[0], b[1]-a[1])
    return direcoes[d]

def isSameDirection(dir1, dir2):
    return abs(int(dir1)-int(dir2)) == 2

def addChainCode(code, newDir, newPoint, pointList):
    if len(code) == 0:
        code += newDir
        pointList.append(newPoint)
    elif isSameDirection(code[-1], newDir):
        code = code[:-1]
        pointList = pointList[:-1]
    else:
        code += newDir
        pointList.append(newPoint)
    
    return code, pointList
    

def chain_code(resample_list, grid_value, chain_type):
    
    assert chain_type == 4 or chain_type == 8
    
    chain_code = ""
    
    resample_list_normalized = []
    for p in resample_list:
        p = (p[0]/grid_value, p[1]/grid_value)
        resample_list_normalized.append(p)
    
    m = 0
    n = 0
    resample_list_normalized = np.array(resample_list_normalized).astype(np.uint16)
    for p in resample_list_normalized:
        if(p[0] > m):
            m = p[0]
        if(p[1] > n):
            n = p[1]
    
    m = int(m)
    n = int(n)
    chainCodeMatrix = np.zeros((m+2,n+2))

    for p in resample_list_normalized:
        chainCodeMatrix[p[0], p[1]] = 1
        
    pil.fromarray(chainCodeMatrix.astype(np.uint8)*255).convert("L").save("chainCodeMatrix.png")
    

    b0 = findBzero(chainCodeMatrix)
    c0 = (b0[0], b0[1] - 1)
    
    b = b0 # no inicio sao iguais
    c = c0
    
    coord_list = []
    coord_list.append(b)
    while(True):
        new_b, c = findNextB_C(chainCodeMatrix, b, c)
        if chain_type == 8:
            chain_code += get8chainDirection(b, new_b)
            coord_list.append(b)
        else:
            if isDiagonal(b, new_b):
                fisrt_code = get4chainDirection(b, c)
                second_code = get4chainDirection(c, new_b)
                chain_code, coord_list = addChainCode(chain_code, fisrt_code, c, coord_list)
                chain_code, coord_list = addChainCode(chain_code, second_code, new_b, coord_list)
            else:
                direcao = get4chainDirection(b, new_b)
                chain_code, coord_list = addChainCode(chain_code, direcao, new_b, coord_list)
        
        b = new_b
        if b in coord_list:
            new_b, new_c = findNextB_C(chainCodeMatrix, b, c)
            if new_b in coord_list:
                break
            
    return chain_code, coord_list, chainCodeMatrix


def normalize_position(chain_code, coor_list):
    start_index = 0
    zeros_count = 0
    begin_zeros_index = -1
    max_zeros_count = 0
    i = 0
    for c in chain_code:
        if c == "0":
            zeros_count += 1
            if begin_zeros_index == -1:
                begin_zeros_index = i
        else:
            if zeros_count > max_zeros_count:
                max_zeros_count = zeros_count
                start_index = begin_zeros_index
            zeros_count = 0
            begin_zeros_index = -1
        i += 1
    
    coor_list_ordenado = coor_list[start_index:] + coor_list[:start_index]
    chain_code_ordenado = chain_code[start_index:] + chain_code[:start_index]
    
    return chain_code_ordenado, coor_list_ordenado

def normalize_rotation(chain_code, chain_type):
    assert chain_type == 4 or chain_type == 8
    
    chain_code_ordenado = ""
    
    diferenca = int(chain_code[0]) - int(chain_code[-1])
    if diferenca < 0:
        diferenca = diferenca + chain_type
        
    chain_code_ordenado += str(diferenca)
    
    for i in range(1, len(chain_code)):
         diferenca = int(chain_code[i]) - int(chain_code[i-1])
         
         if diferenca < 0:
             diferenca = diferenca + chain_type
         chain_code_ordenado += str(diferenca)
    
    return chain_code_ordenado