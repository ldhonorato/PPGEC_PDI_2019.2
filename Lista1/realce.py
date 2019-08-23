import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import math
import hist as h
import os

def realceHistEq(path, alfa, nome):
    img = pil.open(path)
    img = np.array(img)
    w = img.shape[0]
    h = img.shape[1]
    total = w*h
    
    cont = {}
    probCom = {}
        
    for pixel in img.flatten():
        if pixel not in cont.keys():
            cont[pixel] = 1
        else:
            cont[pixel] += 1
    
    for entry in cont.keys():
        cont[entry] /= total
    
    itens = list(cont)
    itens.sort()
    temp = 1
    for x in itens:
        v = 0
        for y in itens[:temp]:
            v += cont[y]
        probCom[x] = int(v*alfa)
        temp += 1
    
    for i in range(w):
        for j in range(h):
            img[i, j] = probCom[img[i, j]]
    
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_hist.png")
        
    
def realceGama(path, alfa, beta, nome):
    img = pil.open(path)
    img = np.array(img).astype(np.float32)/255.
    img = alfa*(img**beta)
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_gama.png")
    
def realceRaiz(path, alfa, nome):
    img = pil.open(path)
    img = alfa*((np.array(img))**0.5)
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_raiz.png")
    
def realceLog(path, alfa, nome):
    img = pil.open(path)
    img = alfa*(np.log10((np.array(img))+1)).astype(np.float32)
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_log.png")

    
def realceLinear(path, alfa, beta, nome):
    img = pil.open(path)
    img = np.array(img)*alfa + beta
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_linear.png")

def f():
    lista_clarear = ["clarear_1", "clarear_2", "clarear_3"]
    
    lista_escurecer = ["escurecer_1", "escurecer_2", "escurecer_3"]
    
    for nome in lista_escurecer:
        path = "images/" + nome + ".png"
        realceHistEq(path, 50, "EqHistograma/"+nome)
        realceGama(path, 220, 2.65, "AjusteGama/"+nome)
        realceRaiz(path, 4, "Raiz/"+nome)
        realceLog(path, 20, "Log/"+nome)
        realceLinear(path, 0.20, 5, "Linear/"+nome)
        
def g():
    lista_pastas = ["AjusteGama/", "EqHistograma/", "Linear/", "Log/", "Raiz/"]
    imagens = os.listdir("images/")[:6]
    pasta_melhor = ["finais/"]
    
    for pasta in pasta_melhor:
        info = os.listdir(pasta)
        cont = 0
        for imagem in imagens:
            h.carrega("images/"+imagem, pasta + info[cont], imagem, info[cont], pasta)
            cont += 1

def main():
    f()
    g()

if __name__ == "__main__":
    main()