#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:57:53 2019

@author: timevisao-mk2
"""
import numpy as np
import queue
import cv2
from bitarray import bitarray

EOF_symbol = 300

class Node:
	def __init__(self):
		self.prob = None
		self.pixelValue = None #apenas folhas possuem valor de pixel
		self.left = None
		self.right = None
	def __lt__(self, other): #métodos para ordenação
		if (self.prob < other.prob): 
			return 1
		else:
			return 0
	def __ge__(self, other):  #métodos para ordenação
		if (self.prob > other.prob):
			return 1
		else:
			return 0

def calculaProbabilidades(img):
    cont = {}
#    total = img.shape[0]*img.shape[1]
        
    for pixel in img.flatten():
        if pixel not in cont.keys():
            cont[pixel] = 1
        else:
            cont[pixel] += 1
    
#    for entry in cont.keys():
#        cont[entry] /= total
    
    return cont

def construirArvore(probabilidades):
    probQueue = queue.PriorityQueue()
    
    for pixelValue in probabilidades:
        node = Node()
        node.pixelValue = pixelValue
        node.prob = float(probabilidades[pixelValue])
#        print(node.prob)
        probQueue.put(node)
    
    while probQueue.qsize() > 1:
        node_l = probQueue.get() #menor probabilidade da fila
        node_r = probQueue.get() #segunda menor probabilidade da fila
        
        newNode = Node()
        newNode.left = node_l
        newNode.right = node_r
        newNode.prob = float(node_l.prob + node_r.prob) #probabilidade é a soma das duas
        probQueue.put(newNode) #removeu os dois ultimos nós e inseriu o novo nó
    
    return probQueue.get() #root node (see -> https://people.ok.ubc.ca/ylucet/DS/Huffman.html)
        

def percorrerArvore(node, codigoPai):
    global dicionarioHuffman
    if node.left is not None:
        novoCodigo = codigoPai + '0'
        percorrerArvore(node.left, novoCodigo)
    
    if node.right is not None:
        novoCodigo = codigoPai + '1'
        percorrerArvore(node.right, novoCodigo)
    
    if node.left == None and node.right == None: #é uma folha
        dicionarioHuffman[node.pixelValue] = codigoPai
#        wr_str = str(node.pixelValue)+'->'+ codigoPai+'\n'
#        print(wr_str)
#        f.write(wr_str)

def generate_reverse_dict(dict_in):
    dict_reverso = {}
    for k in dict_in:
        dict_reverso[dict_in[k]] = k
    return dict_reverso

dicionarioHuffman = {} #dicionario global
def encode_huffman(path, prefix):
    img = cv2.imread(path,0)
    probabilidades = calculaProbabilidades(img)
    probabilidades[EOF_symbol] = 1
    root_node = construirArvore(probabilidades)
    percorrerArvore(root_node,'')#percorre a árvore e preenche o dicionário global
    
    #salva o dicionário em txt
    f = open(prefix + '_dic_huffman.txt','w')
    for valorPixel in dicionarioHuffman:
        wr_str = str(valorPixel)+'->'+ dicionarioHuffman[valorPixel]+'\n'
        f.write(wr_str)
    f.close()
    
    #salva a imagem em txt e em binário
    #arquivo em txt é muito grande - abrir com o nano
    f_txt = open(prefix + '_codificadaHuffman.txt', 'w')
    encoded = bitarray()
    quantidade_bits = 0
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            code = dicionarioHuffman[img[i, j]]
            encoded.extend(bitarray(code))
            quantidade_bits += len(code)
            f_txt.write(code + ' ')
        f_txt.write('\n')
    
    f_txt.close()
    
    if len(encoded)%8 != 0:
        code = dicionarioHuffman[EOF_symbol]
        quantidade_bits += len(code)
        encoded.extend(bitarray(code))
        bits_faltantdo = len(encoded)%8
        print('completando: ')
        print('EOF', len(code))
        print('pad', bits_faltantdo)
        encoded.extend(bitarray(str().zfill(bits_faltantdo)))
        quantidade_bits += bits_faltantdo
    
    f_bin = open(prefix + '_codificadaHuffman.bin', 'wb')
    encoded.tofile(f_bin)
    f_bin.close()
    
    #salva o dicionário - pode ser salvo como um cabeçalho do arquivo
    f_bin = open(prefix + '_huffman_header.bin', 'wb')
    f_bin.write(img.shape[0].to_bytes(2, byteorder='big'))
    f_bin.write(img.shape[1].to_bytes(2, byteorder='big'))
    for k, v in probabilidades.items():
        if k == EOF_symbol:
            k = 0
        f_bin.write(int(k).to_bytes(1, byteorder='big'))
        f_bin.write(int(v).to_bytes(3, byteorder='big'))
    f_bin.close()
    
    return quantidade_bits

def decode_huffman(path_imagem, path_dicionario):
    probabilidades = {}
    global dicionarioHuffman
    
    with open(path_dicionario, "rb") as f:
        
        dim1 = int.from_bytes(f.read(2), "big") #dimensão 1
        dim2 = int.from_bytes(f.read(2), "big") #dimensão 2
              
        encontrou_EOF = False
        while not encontrou_EOF:
            key = int.from_bytes(f.read(1), "big")
            value = int.from_bytes(f.read(3), "big")
            if key == 0 and key in probabilidades:
                key = EOF_symbol
                encontrou_EOF = True
            probabilidades[key] = value
                
    root_node = construirArvore(probabilidades)
    
    dicionarioHuffman = {} #limpa o dicionario
    percorrerArvore(root_node,'')#percorre a árvore e preenche o dicionário global
    
    dicionarioHuffman_reverso = generate_reverse_dict(dicionarioHuffman)
    img_reconstruida = np.zeros(dim1*dim2)
    
    read_encoded = bitarray()
    f_bin = open(path_imagem, 'rb')
    read_encoded.fromfile(f_bin)
    f_bin.close()
    
    start_index = 0
    end_index = 1
    
    code = read_encoded[start_index:end_index].to01()
    for i in range(img_reconstruida.shape[0]):
        while code not in dicionarioHuffman_reverso:
            end_index += 1
            code = read_encoded[start_index:end_index].to01()
        
        simbolo = dicionarioHuffman_reverso[code]
        if simbolo != EOF_symbol:
            img_reconstruida[i]= simbolo
            start_index = end_index
            end_index += 1
            code = read_encoded[start_index:end_index].to01()
        else:
            break
    
    img_reconstruida = img_reconstruida.reshape(dim1, dim2)
    return img_reconstruida
    
if __name__ == "__main__":
    paths = ['images/image_1.tif', 'images/image_2.tif', 'images/image_3.tif']
    
    for path in paths:
        prefix = path.split('/')[1][:-4]
    #    path = 'image_1_resize.tif'
    #    prefix = 'imagem_1_resize'
        print('##############################################')
        print(path)
        img = cv2.imread(path, 0)
        print(img.shape)
        
        qtde_bits_compressao = encode_huffman(path, prefix)
        img_reconstruida = decode_huffman(prefix+'_codificadaHuffman.bin', prefix+'_huffman_header.bin')
               
        qtde_bits_sem_compressao = img.shape[0]*img.shape[1]*8
        
        print('Imagens são iguais:', np.array_equal(img, img_reconstruida))
        
        print('##############################################')
        print("Tamanho sem Compressao", qtde_bits_sem_compressao)
        print("Tamanho codigo Huffman", qtde_bits_compressao)
        tamanho_dicionario = 4 + (4*len(dicionarioHuffman))    
        print("Tamanho dicionario", tamanho_dicionario)
        qtde_bits_compressao = qtde_bits_compressao + tamanho_dicionario
        print("Tamanho codigo Huffman com dic", qtde_bits_compressao)
        print('---------------------------------')
        
        taxa_compressao = qtde_bits_sem_compressao/qtde_bits_compressao
        print('Taxa de compressao = ', taxa_compressao)
        print('Redundância relativa = ', 1-(1/taxa_compressao))
    
