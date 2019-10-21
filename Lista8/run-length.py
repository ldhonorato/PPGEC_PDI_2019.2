#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:57:53 2019

@author: timevisao-mk2
"""
import numpy as np
import cv2


def run_lenght_code(img, endLine=False):
    list_of_run_lenght = []
    if not endLine:
        list_of_run_lenght.append(img.shape[0])
        list_of_run_lenght.append(img.shape[1])    
        
        img = img.flatten()
        previous_simbolo = current_simbolo =  img[0]
        qtde_simbolos = 1
        
        for i in range(1, img.shape[0]):
            current_simbolo = img[i]
            if current_simbolo == previous_simbolo:
                qtde_simbolos += 1
                                
                if qtde_simbolos == 256: #maximum value - one byte
                    list_of_run_lenght.append(qtde_simbolos-1)
                    list_of_run_lenght.append(current_simbolo)
                    previous_simbolo = current_simbolo
                    qtde_simbolos = 1
            else:
                list_of_run_lenght.append(qtde_simbolos)
                list_of_run_lenght.append(previous_simbolo)
                previous_simbolo = current_simbolo
                qtde_simbolos = 1
                
        #adiciona os ultimos bytes da imagem
        list_of_run_lenght.append(qtde_simbolos)
        list_of_run_lenght.append(current_simbolo)
        
    else:    
        for i in range(img.shape[0]):
            previous_simbolo = current_simbolo =  img[i,0]
            qtde_simbolos = 1
            flag_adicionada = False
            for j in range(1, img.shape[1]):
                current_simbolo = img[i,j]
                if current_simbolo == previous_simbolo:
                    qtde_simbolos += 1
                    
                    if qtde_simbolos == 255: #maximum for one byte
                        list_of_run_lenght.append(qtde_simbolos)
                        list_of_run_lenght.append(current_simbolo)
                        previous_simbolo = current_simbolo
                        qtde_simbolos = 1
                        flag_adicionada = True
                    else:
                        flag_adicionada = False
                else:
                    list_of_run_lenght.append(qtde_simbolos)
                    list_of_run_lenght.append(current_simbolo)
                    previous_simbolo = current_simbolo
                    qtde_simbolos = 1
                    flag_adicionada = True
            
            if flag_adicionada == False:
                list_of_run_lenght.append(qtde_simbolos)
                list_of_run_lenght.append(current_simbolo)
            
            list_of_run_lenght.append(0) #end of line
            list_of_run_lenght.append(1) #end of line
    
        
    #list_of_run_lenght.append((0, 1)) #end of image
    return list_of_run_lenght


def agrupar_run_lenght_code(list_of_run_lenght, hasEndLine):
#    optimizes_list_of_run_lenght = []
    codigo_agrupado = []
    lista_atual = []
    if hasEndLine:
        indice_inicio = 2
    else:
        codigo_agrupado.append(list_of_run_lenght[0])
        codigo_agrupado.append(list_of_run_lenght[1])
        indice_inicio = 4
    
    lista_atual.append(list_of_run_lenght[indice_inicio-2])
    lista_atual.append(list_of_run_lenght[indice_inicio-1])
    
#    flag_lista_atual_adicionada = False
    for i in range(indice_inicio, len(list_of_run_lenght), 2):
        if list_of_run_lenght[i] != 0:
            if lista_atual[0] == list_of_run_lenght[i]:
                lista_atual.append(list_of_run_lenght[i])
                lista_atual.append(list_of_run_lenght[i+1])
#                flag_lista_atual_adicionada = False
            else:
                codigo_agrupado.append(lista_atual)
#                flag_lista_atual_adicionada = False
                lista_atual = []
                lista_atual.append(list_of_run_lenght[i])
                lista_atual.append(list_of_run_lenght[i+1])
                
    
#    if flag_lista_atual_adicionada == False:
    codigo_agrupado.append(lista_atual)
    
    return codigo_agrupado

def otimizar_run_lenght_code(codigo_agrupado, hasEndLine):
    codigo_otimizado = []
    if hasEndLine:
        indice_inicio = 0
    else:
        codigo_otimizado.append(codigo_agrupado[0])
        codigo_otimizado.append(codigo_agrupado[1])
        indice_inicio = 2
    
    
    for i in range(indice_inicio, len(codigo_agrupado)):
        lista = codigo_agrupado[i]
        if lista[0] == 1 and len(lista) > 4:
            codigo_otimizado.append(0)
            qtde_inteiros = int(len(lista)/2)
            codigo_otimizado.append(qtde_inteiros)
            for j in range(1, len(lista), 2):
                codigo_otimizado.append(lista[j])
        else:
            for j in range(len(lista)):
                codigo_otimizado.append(lista[j])
    
    return codigo_otimizado
        

def save_file(code, fileName):
    f = open(fileName, 'wb')
    
    f.write(code[0].to_bytes(2, byteorder='big'))
    f.write(code[1].to_bytes(2, byteorder='big'))
    
    for i in range(2, len(code)):
        f.write(int(code[i]).to_bytes(1, byteorder='big'))
    
    f.close()

def read_file(fileName, hasEndLine):
    code = []
    if not hasEndLine:
        with open(fileName, "rb") as f:
            dim1 = f.read(2) #dimensão 1
            dim2 = f.read(2) #dimensão 2
            code.append(int.from_bytes(dim1, "big"))
            code.append(int.from_bytes(dim2, "big"))
            
            byte = f.read(1)
            while byte:
                # Do stuff with byte.
                code.append(int.from_bytes(byte, "big"))
                byte = f.read(1)
    
        y = code.pop(0)
        x = code.pop(0)
        img_reconstruida = np.zeros(y*x)
        img_index = 0
        while img_index != img_reconstruida.shape[0]:
            c = code.pop(0)
            if c == 0:
                qtde = code.pop(0)
                while qtde > 0:
                    byte = code.pop(0)
                    img_reconstruida[img_index] = byte
                    img_index += 1
                    qtde -= 1
            else:
                qtde = c
                byte = code.pop(0)
                while qtde > 0:
                    img_reconstruida[img_index] = byte
                    img_index += 1
                    qtde -= 1
        img_reconstruida = img_reconstruida.reshape(y, x)
        
        return img_reconstruida
    
    return np.zeros(0)
                

if __name__ == "__main__":
    paths = ['images/image_1.tif', 'images/image_2.tif', 'images/image_3.tif']
    
    for path in paths:
        prefix = path.split('/')[1][:-4]
        #codifica
        img = cv2.imread(path,0)

        rl_code = run_lenght_code(img, endLine=False)
        
        codigo_agrupado = agrupar_run_lenght_code(rl_code, hasEndLine=False)
        run_lenght_code_otimizado = otimizar_run_lenght_code(codigo_agrupado, hasEndLine=False)
        
        save_file(run_lenght_code_otimizado, prefix + '_RLC_otimizado.bin')
        imagem_recuperada = read_file(prefix + '_RLC_otimizado.bin', False)
        
        print('##############################################')
        print(path)
        print(img.shape)
               
        print('Imagens são iguais:', np.array_equal(img, imagem_recuperada))
        
        qtde_bits_sem_compressao = img.shape[0]*img.shape[1]*8
        
        qtde_bits_compressao = len(rl_code)*8
        qtde_bits_compressao_otimizado = len(run_lenght_code_otimizado)*8
        
        print("Tamanho sem Compressao", qtde_bits_sem_compressao)
        print("Tamanho com Compressao", qtde_bits_compressao)
        print("Tamanho com Compressao Otimizada", qtde_bits_compressao_otimizado)
        print('---------------------------------')
        taxa_compressao = qtde_bits_sem_compressao/qtde_bits_compressao
        print('Taxa de compressao = ', taxa_compressao)
        print('Redundância relativa = ', 1-(1/taxa_compressao))
        print('---------------------------------')
        taxa_compressao = qtde_bits_sem_compressao/qtde_bits_compressao_otimizado
        print('Taxa de compressao otimizada= ', qtde_bits_sem_compressao/qtde_bits_compressao_otimizado)
        print('Redundância relativa otimizada = ', 1-(1/taxa_compressao))
        print('##############################################')

#f = open('image_1_run_code.txt', 'w+')


