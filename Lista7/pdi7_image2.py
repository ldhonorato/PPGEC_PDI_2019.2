#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:12:28 2019

@author: timevisao-mk2
"""
import numpy as np
import PIL.Image as pil
import chain_codes as chain
import cv2

def label_vertices(chain_code, points):
    dic_vertices = {"01":'P',
                    "03":'B',
                    "10":'B',
                    "12":'P',
                    "21":'B',
                    "23":'P',
                    "30":'P',
                    "32":'B'}
    
    vertices = []
    vertices.append((points[0], 'B'))
    for i in range(len(chain_code)-1):
        code = chain_code[i:i+2]
        try:
            tipoVertice = dic_vertices[code]
            ponto = points[i+1]
            
            if tipoVertice == 'P':
                if code == "01":
                    ponto = (ponto[0]-1, ponto[1]-1)
                elif code == "12":
                    ponto = (ponto[0]+1, ponto[1]-1)
                elif code == "23":
                    ponto = (ponto[0]+1, ponto[1]+1)
                else: #code == "30"
                    ponto = (ponto[0]-1, ponto[1]+1)
            
            vertices.append((ponto, tipoVertice))
        except:
            pass
    
    return vertices            

def sinal(p1, p2, p3):
    p1 = p1[0]
    p2 = p2[0]
    p3 = p3[0]
    matriz = np.array([[p1[0], p1[1], 1],
                       [p2[0], p2[1], 1],
                       [p3[0], p3[1], 1]])
    
    return np.linalg.det(matriz)
    

def mpp(vertices):
    vertices_mpp = []
    v0 = vertices[0]
    Br = Pr =  v0
    
    vertices_mpp.append(v0)
    
    for i in range(1, len(vertices)):
        Vk = vertices[i]
        Vl = vertices_mpp[-1]
        
        if sinal(Vl, Br, Vk) > 0:
            vertices_mpp.append(Br)
            Pr = Br
        elif sinal(Vl, Pr, Vk) >= 0:
            if Vk[1] == 'B':
                Br = Vk
            else:
                Pr = Vk
        else: #sinal(Vl, Pr, Vk) < 0:
            vertices_mpp.append(Pr)
            Br = Pr
    
    
    Vk = vertices[0]
    Vl = vertices_mpp[-1]
    
    if sinal(Vl, Br, Vk) > 0:
        vertices_mpp.append(Br)
    elif sinal(Vl, Pr, Vk) < 0:
        vertices_mpp.append(Pr)
    
    return vertices_mpp


def encontrarPontoProximo(p1, p2):
    (y, x) = p1
    pontos = [(y-1,x-1), (y-1, x), (y-1, x+1), 
              (y, x-1), p1, (y, x+1),
              (y+1, x-1), (y+1, x),(y+1, x+1)]
    
    nextP = p1
    distanciaMinima = float('inf')
    for p in pontos:
        current_distance = chain.distance(p, p2)
        if current_distance < distanciaMinima:
            distanciaMinima = current_distance
            nextP = p
    
    return nextP


def desenharPoligono(vertices_mpp, imageShape):
    imagemPreenchida = np.zeros(imageShape)
    
    for i in range(len(vertices_mpp)-1):
        ponto1 = vertices_mpp[i][0]
        ponto2 = vertices_mpp[i+1][0]
        
        imagemPreenchida[ponto1] = 255
        while ponto1 != ponto2:
            ponto1 = encontrarPontoProximo(ponto1, ponto2)
            imagemPreenchida[ponto1] = 255
        
    
    ponto1 = vertices_mpp[-1][0]
    ponto2 = vertices_mpp[0][0]
    
    imagemPreenchida[ponto1] = 255
    while ponto1 != ponto2:
        ponto1 = encontrarPontoProximo(ponto1, ponto2)
        imagemPreenchida[ponto1] = 255
    
    return imagemPreenchida


if __name__ == "__main__":
    grid_value = 10
    min_dist = 6
    run_all = True
    
    if run_all:
        img2 = cv2.imread('images/image_2.png', 0)
        img2 = img2 / np.max(img2)
        img2 = np.logical_not(img2)*255
        pil.fromarray(img2.astype(np.uint8)).convert("L").save("not_image_2.png")
        
        border_img, coord_list = chain.fronteira("not_image_2.png", "border_image2.png")
        resample_img, resample_list, grid = chain.resample(border_img, coord_list, min_dist, grid_value)
            
        grid_or_border = np.logical_or(grid, border_img)*255
        pil.fromarray(grid_or_border.astype(np.uint8)).convert("L").save("grid_or_border_img.png")
        
        x = grid + border_img + resample_img
        x = (x/np.max(x))*255
        pil.fromarray(x.astype(np.uint8)).convert("L").save("x.png")
        
        pil.fromarray(resample_img*255).convert("L").save("resample_border_img.png")
        pil.fromarray(grid*255).convert("L").save("grid.png")
    else:
        resample_img = cv2.imread('resample_border_img.png', 0)/255
        resample_list = np.where(resample_img == 1)
        resample_list = list(zip(resample_list[0], resample_list[1]))
    
    chain_code4, points_c_code_4, chainCodeMatrix_4 = chain.chain_code(resample_list, grid_value, chain_type=4)
    
    for i in range(len(points_c_code_4)):
        points_c_code_4[i] = (points_c_code_4[i][0]*grid_value, points_c_code_4[i][1]*grid_value)
        
    pontos_rotulados = label_vertices(chain_code4, points_c_code_4)
    
#    pontos_rotulados_reordenados = []
#    
#    pontos_rotulados_reordenados.append(pontos_rotulados[0])
#    
#    for i in range(1, len(pontos_rotulados)):
#        pontos_rotulados_reordenados.append(pontos_rotulados[-i])
#    
    
    
    vertices_mpp = mpp(pontos_rotulados)
    
    img_vertices_mpp = np.zeros((542, 636))
    for p in vertices_mpp:
        img_vertices_mpp[p[0]] = 255
    
    pil.fromarray(img_vertices_mpp).convert("L").save("vertices_mpp.png")
    
    imagem_completa = desenharPoligono(vertices_mpp, (542, 636))
    pil.fromarray(imagem_completa).convert("L").save("poligono_mpp.png")
    
#    for p in pontos_rotulados:
#        valor = 0
#        if p[1] == 'B':
#            valor = 10
#        else:
#            valor = 20
#        chainCodeMatrix_4[p[0]] = valor

     
#    for p in vertices_mpp:
#        chainCodeMatrix_4[p[0]] = 50
        
    