#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:46:04 2019

@author: timevisao-mk1
"""

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2

def findBzero(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                b0 = (i, j)
                return b0

def findNextC(img, b):
    coord_list = [(b[0]-1, b[1]), (b[0], b[1]+1), 
                  (b[0]+1, b[1]), (b[0], b[1]-1)]
    for coord in coord_list:
        if img[coord[0], coord[1]] == 0:
            return coord

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5            
            
def findNextB(img, b, c):
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
            return coord
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
        b = findNextB(img, b, c)
        c = findNextC(img, b)
        
        if b in coord_list:
            break
    
    pil.fromarray(newImg*255).convert("L").save(nome)
    
    return newImg, coord_list

def fronteira2(img):
    
    b0 = findBzero(img)
    c0 = (b0[0], b0[1] - 1)
    
    b = b0 # no inicio sao iguais
    c = c0
    
    coord_list = []
    newImg = np.zeros(img.shape)
        
    while(True):
        coord_list.append(b)
        newImg[b[0], b[1]] = 1
        b = findNextB(img, b, c)
        c = findNextC(img, b)
        
        if b in coord_list:
            break
        
    return newImg, coord_list

def coord_ones_list(img):
#   img = cv2.imread(path, 0)/255
   ones_list = []
   
   for i in range(img.shape[0]):
       for j in range(img.shape[1]):
           if img[i, j] == 1:
               ones_list.append((i, j))
   return ones_list


#def MAT(region_list, border_list, border_img, nome):
#    newImg = np.zeros(border_img.shape)
#    
#    medial_axis = []
#    
#    for r in region_list:
#        min_dist = float("inf")
#        min_coord = None
#                
#        for b in border_list:
#            if b != r:    
#                dist = distance(r, b)
#                if dist <= min_dist:
#                    min_dist = dist
#                    min_coord = b
#        print(r, min_coord, min_dist)
#        
#        for b in border_list:
#            dist = distance(r, b)
#            if dist == min_dist and b != min_coord:
#                newImg[r] = 1
#                
#        medial_axis.append(r)
#    pil.fromarray(newImg*255).convert("L").save(nome)
#    pil.fromarray(np.logical_or(newImg, border_img).astype(np.int8)*255).convert("L").save("or_"+nome)
#    
#    return newImg, medial_axis, np.logical_or(newImg, border_img)

def verifica_passo1(subImg):
    [[a, b, c],
     [h, _, d],
     [g, f, e]] = subImg
           
    N = a + b + c + d + e + f + g + h
    
    if N < 2 or N > 6:
        return False
    
    T = 0
    lista = [b,c,d,e,f,g,h,a]
    
    for temp in range(7):
        if lista[temp] == 0 and lista[temp+1] == 1:
                T += 1
    
    if lista[7] == 0 and lista[0] == 1:
        T += 1

    if T != 1:
        return False
    
    produto_c = b*d*f
    produto_d = d*f*h

    if produto_c == 0 and produto_d == 0:
        return True
    
    return False

def verifica_passo2(subImg):
    [[a, b, c],
     [h, _, d],
     [g, f, e]] = subImg
           
    N = a + b + c + d + e + f + g + h
    
    if N < 2 or N > 6:
        return False
    
    T = 0
    lista = [b,c,d,e,f,g,h,a]
    
    for temp in range(7):
        if lista[temp] == 0 and lista[temp+1] == 1:
                T += 1
    
    if lista[7] == 0 and lista[0] == 1:
        T += 1

    if T != 1:
        return False
    
    produto_c_linha = b*d*h
    produto_d_linha = b*f*h
    
    if produto_c_linha == 0 and produto_d_linha == 0:
        return True
    
    return False


def passo1(subImg):
    [[a, b, c],
     [h, _, d],
     [g, f, e]] = subImg
     
    produto_c = b*d*f
    produto_d = d*f*h
    
    return verifica(subImg) and produto_c == 0 and produto_d == 0

def passo2(subImg):
    [[a, b, c],
     [h, _, d],
     [g, f, e]] = subImg
     
    produto_c = b*d*h
    produto_d = b*f*h
    
    return verifica(subImg) and produto_c == 0 and produto_d == 0

     
def afinamento(img, border_list):
    for p in border_list:
        if img[p] == 1:
            subImg = img[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
            if passo1(subImg) and passo2(subImg):
                img[p] = 0
    
    return img
    
    
def MAT(path, nome):
    img = cv2.imread(path, 0)/255
    
#    newImg = np.zeros(img.shape)
#    coord_img_ones = coord_ones_list(img)
    
    points_to_remove = []
    flag_run = True
    iteracoes = 0
    pontos_removidos = 0
    while flag_run:
        # passo 1
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                
                if img[i, j] == 1:
                    subImg = img[i-1:i+2, j-1:j+2]
                    if verifica_passo1(subImg):
                        points_to_remove.append((i,j))
        
        pontos_removidos = len(points_to_remove)
        for p in points_to_remove:
            img[p] = 0
        points_to_remove.clear()
        
        
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                
                if img[i, j] == 1:
                    subImg = img[i-1:i+2, j-1:j+2]
                    if verifica_passo2(subImg):
                        points_to_remove.append((i,j))
        
        pontos_removidos += len(points_to_remove)
        for p in points_to_remove:
            img[p] = 0
        points_to_remove.clear()
    
        iteracoes += 1
        print("Iteracao = {iter} Pontos removidos = {remove}".format(iter=iteracoes, remove=pontos_removidos))
            
        if iteracoes % 20 == 0:
            pil.fromarray(img*255).convert("L").save(str(iteracoes) +'_' + nome)
        
        if pontos_removidos == 0:
            flag_run = False
        
    pil.fromarray(img*255).convert("L").save(nome)
    
    return img

if __name__ == "__main__":
#    border_img, coord_list = fronteira("images/image_3.png", "border_image3.png")
#    orig_ones_list = coord_ones_list("images/image_3.png")
#    medial_axis, coord_medial_axis, or_medial_axis = MAT(orig_ones_list, coord_list, border_img, "esqueleto.png")
#    a = afinamento("image_3.png", coord_list ,"teste.png")
#    border_img2, coord_list2 = fronteira("teste.png", "teste2.png")
    
    img = MAT("images/image_3.png", "algumacoisa.png")




















