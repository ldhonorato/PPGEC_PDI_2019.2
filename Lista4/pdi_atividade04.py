# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:11:48 2019

@author: Agostinho
"""

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import random
import hist
import math

def mediaHarmonica(subImg):
    d = 0
    for i in range(subImg.shape[0]):
        for j in range(subImg.shape[0]):
            d += 1/subImg[i,j]
    
    return (subImg.shape[0]*subImg.shape[1])/d

def statistic_filter(path, kernel_size, statistic_function, name):
    img = pil.open(path)
    img = np.array(img)
    padding_size = kernel_size - 2
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    for i in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            subImg = padded_img[ni-padding_size:ni +
                (padding_size+1), nj-padding_size:nj+(padding_size+1)]
            newImg[i, j] = statistic_function(subImg)
    
    newImg = pil.fromarray(newImg.astype(np.uint8))
    newImg = newImg.convert("L")
    newImg.save(name)

    return newImg

def adaptative_filter(path, kernel_size, noise_var, name):
    img = pil.open(path)
    img = np.array(img)
    padding_size = kernel_size - 2
    padded_img, original_shape = padding(img, padding_size)
    newImg = np.zeros(original_shape)

    central_pixel = int(kernel_size/2)
    for i in range(padded_img.shape[0] - padding_size*2):
        for j in range(padded_img.shape[1] - padding_size*2):
            ni = i + padding_size
            nj = j + padding_size
            subImg = padded_img[ni-padding_size:ni+(padding_size+1), nj-padding_size:nj+(padding_size+1)]

            var_rate = noise_var/np.var(subImg)
            if var_rate > 1:
                var_rate = 1
            
            newImg[i, j] = subImg[central_pixel, central_pixel] - var_rate*(subImg[central_pixel, central_pixel]-np.mean(subImg))
    
    newImg = pil.fromarray(newImg)
    newImg = newImg.convert("L")
    newImg.save(name)

    return newImg


def padding2(img, valor=0): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]*2, img.shape[1]*2)) + valor
    newImg[0:img.shape[0], 0:img.shape[1]] = img

    return newImg

def removePadding2(img, originalShape):
    return img[0:originalShape[0], 0:originalShape[1]]

def realceGama(path, alfa, beta, nome):
    img = pil.open(path)
    img = np.array(img).astype(np.float32)/255.
    img = alfa*(img**beta)
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome + "_gama.png")

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

def padding(img, camadas=1, valor=255): #caso queira uma borda com valor diferente
    newImg = np.zeros((img.shape[0]+2*camadas, img.shape[1]+2*camadas)) + valor
    limx = newImg.shape[0]
    limy = newImg.shape[1]
    newImg[camadas:limx-camadas, camadas:limy-camadas] = img
    
    return newImg, img.shape


def gauss(subImg, media, desvio, dim):
        
    g = np.linspace(media-3*desvio, media+3*desvio, dim**2)
    
    #print(g)

    p = np.exp(-((g - media)**2)/(2*desvio**2))/(2*np.pi*desvio)**0.5
    p = p.reshape((dim, dim))
    
    #print(p)
    p = p/np.sum(p)
    
    pixel = np.sum(subImg*p)
    
    return pixel, p

def fGauss(imagem, media, desvio, dim):
    valor = int((dim - 1)/2)    
    img, origShape = padding(imagem, valor, 0)
    newImg = np.zeros(origShape)
          
    for i  in range(img.shape[0] - valor*2):
        for j in range(img.shape[1] - valor*2):
            ni = i + valor
            nj = j + valor
            newImg[i, j], filtro = gauss(img[ni-valor:ni+valor+1,nj-valor:nj+valor+1], media, desvio, dim) #sub matriz
    
    return newImg, filtro

def mediana(subImg):
    filtro = subImg.flatten()
    filtro.sort()
    return filtro[int(len(filtro)/2)]


def fMediana(imagem):
    img, origShape = padding(imagem)
    newImg = np.zeros(origShape)
      
    for i  in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            ni = i + 1
            nj = j + 1
            newImg[i, j] = mediana(img[ni-1:ni+2,nj-1:nj+2]) #sub matriz
    
    return newImg

def filtro(path, function, repeticoes, nome):
    img = pil.open(path)
    img = np.array(img)
    
    for rep in range(repeticoes):
        img = function(img)
    
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome)

def filtro2(path, function, media, desvio, dim, repeticoes, nome):
    img = pil.open(path)
    img = np.array(img)
    
    for rep in range(repeticoes):
        img, filtro = function(img, media, desvio, dim)
    
    img = pil.fromarray(img)
    img = img.convert("L")
    img.save(nome)
    return filtro

def IMP_UNI(path, nome):
    img = pil.open(path)
    img = np.array(img)
        
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.randint(0,19) in [0,1,2]:
                img[i, j] = 0
    
    im = pil.fromarray(img)
    im = im.convert("L")
    im.save(nome)

def impulso_unipolar(path, prob, value, name):
    img = pil.open(path)
    img = np.array(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd <= prob:
                img[i, j] = value
    
    im = pil.fromarray(img)
    im = im.convert("L")
    im.save(name)

def impulso_biipolar(path, prob, probSalt, name, valueSalt=255, valuePepper=0):
    img = pil.open(path)
    img = np.array(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd <= prob:
                saltOrPepper_prob = random.random()
                if(saltOrPepper_prob > probSalt):
                    img[i, j] = valueSalt
                else:
                    img[i, j] = valuePepper
    
    im = pil.fromarray(img)
    im = im.convert("L")
    im.save(name)
    
def gaussian_noise(path, mean, std, name):
    img = pil.open(path)
    img = np.array(img)

    noise = np.random.normal(mean, std, img.shape)

    g = img + noise

    noise = pil.fromarray(noise.astype(np.uint8))
    noise = noise.convert('L')
    noise.save('Gaussian noise_m=' + str(mean) + '_si=' + str(std)+ '.png')

    im = pil.fromarray(g.astype(np.uint8))
    im = im.convert("L")
    im.save(name)

def IMP_BIP(path, nome):
    img = pil.open(path)
    img = np.array(img)
    
    nv = [0,255]
        
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.randint(0,9) == 0:
                img[i, j] = nv[random.randint(0,1)]
    
    im = pil.fromarray(img)
    im = im.convert("L")
    im.save(nome)

def GAUSS(path, nome):
    img = pil.open(path)
    img = np.array(img)  
    for k in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                    img[i, j, k] += np.random.normal(loc=15, scale=10)
    
    im = pil.fromarray(np.round(img))
    im = im.convert("L")
    im.save(nome)

def filtroMedianaAdaptativo(path, wi, wmax, name):
    img = pil.open(path)
    img = np.array(img)

    valor_pad = int((wmax - 1)/2)    
    img, origShape = padding(img, valor_pad, 0)
    newImg = np.zeros(origShape)
    

    for i  in range(img.shape[0] - valor_pad*2):
        print(i)
        for j in range(img.shape[1] - valor_pad*2):
            ni = i + valor_pad
            nj = j + valor_pad
            janela_atual = wi
            pad_atual =int((janela_atual - 1)/2)    
            flagPixelOk = False
            while flagPixelOk == False:
                zmin = np.min(img[ni-pad_atual:ni+pad_atual+1,nj-pad_atual:nj+pad_atual+1])
                zmax = np.max(img[ni-pad_atual:ni+pad_atual+1,nj-pad_atual:nj+pad_atual+1])
                zmed = np.median(img[ni-pad_atual:ni+pad_atual+1,nj-pad_atual:nj+pad_atual+1])

                A1 = zmed - zmin
                A2 = zmed - zmax
                if A1>0 and A2<0:
                    B1 = img[ni, nj] - zmin
                    B2 = img[ni, nj] - zmax
                    if B1>0 and B2<0:
                        newImg[i,j] = img[ni, nj]
                    else:
                        newImg[i,j] = zmed
                    flagPixelOk = True
                else:
                    janela_atual += 2
                    if janela_atual > wmax:
                        newImg[i,j] = zmed
                        flagPixelOk = True
    
    newImg = pil.fromarray(np.round(newImg))
    newImg = newImg.convert("L")
    newImg.save(name)
    return newImg

def imagem05(path, nome):
    img = np.array(pil.open(path))
    originalShape = img.shape
    img = padding2(img)
    img = img/255.

    imgfft = np.fft.fft2(img)
    imgfftshift = np.fft.fftshift(imgfft)
    spectro = 20*np.log(np.abs(imgfftshift))

    
    noise = np.random.normal(0, 10, img.shape)
    noisefft = np.fft.fft2(noise)
    noisefftshift = np.fft.fftshift(noisefft)

    noise_spectro = 20*np.log(np.abs(noisefftshift))
    
    dif_fourier = imgfftshift - noisefftshift

    dif_spectro = 20*np.log(np.abs(dif_fourier))

    dif_fourier = np.fft.ifftshift(dif_fourier)
    newImg = np.fft.ifft2(dif_fourier)
    newImg = np.abs(newImg)

    newImg = removePadding2(newImg, originalShape)
    
    noise = pil.fromarray(noise.astype(np.uint8))
    noise = noise.convert('L')
    noise.save('Gaussian noise_m=0_si=10.png')

    newImg = pil.fromarray(newImg*255.)
    newImg = newImg.convert('L')
    newImg.save(nome)

    spectro = pil.fromarray(spectro.astype(np.uint8))
    spectro = spectro.convert('L')
    spectro.save('spectro_' + nome)

    noise_spectro = pil.fromarray(noise_spectro.astype(np.uint8))
    noise_spectro = noise_spectro.convert('L')
    noise_spectro.save('spectro_Gaussian noise_m=0_si=10.png')

    dif_spectro = pil.fromarray(dif_spectro.astype(np.uint8))
    dif_spectro = dif_spectro.convert('L')
    dif_spectro.save('spectro_diferenca.png')



if __name__=="__main__":
    #---------------------------------------------------
    #----------------Imagem 01 -------------------------
    #---------------------------------------------------
    impulso_unipolar("images/image_01.png", 0.15, 0, "image_01_uni_15per.png")
    hist.carrega("images/image_01.png", "image_01_uni_15per.png", "image_01", "image_01_uni_15per")
    
    #---------------------------------------------------
    #----------------Imagem 02 -------------------------
    #---------------------------------------------------
    impulso_biipolar("images/image_02.png", 0.1, 0.5, "image_02_bip_10per.png")
    hist.carrega("images/image_02.png", "image_02_bip_10per.png", "image_02", "image_02_bip_10per")

    #---------------------------------------------------
    #----------------Imagem 03 -------------------------
    #---------------------------------------------------
    gaussian_noise("images/image_03.png", 15, 10, 'image_03_gaussianNoise.png')
    hist.carrega("images/image_03.png", "image_03_gaussianNoise.png", "image_03", "image_03_gaussianNoise")
    
    #---------------------------------------------------
    #----------------Imagem 04 -------------------------
    #---------------------------------------------------
    filtro("images/image_04.png", fMediana, 1, "image_04_mediana1.png")
    hist.carrega("images/image_04.png", "image_04_mediana1.png", "image_04", "image_04_mediana x1")

    filtroMedianaAdaptativo("images/image_04.png", 3, 7, "image_04_medianaAdaptativo.png")
    hist.carrega("images/image_04.png", "image_04_medianaAdaptativo.png", "image_04", "image_04_medianaAdaptativo")

    image4 = np.array(pil.open("images/image_04.png"))
    image4_x1 = np.array(pil.open("image_04_mediana1.png"))
    image4_x2 = np.array(pil.open("image_04_mediana2.png"))
    image4_x3 = np.array(pil.open("image_04_mediana2.png"))
    
    hist.histograma([image4 - image4_x1], 1, ["Ruido - image_04_gaussx1"], "Ruido - image_04_gaussx1")
    hist.histograma([image4 - image4_x2], 1, ["Ruido - image_04_gaussx2"], "Ruido - image_04_gaussx2")
    hist.histograma([image4 - image4_x3], 1, ["Ruido - image_04_gaussx3"], "Ruido - image_04_gaussx3")
    
    #---------------------------------------------------
    #----------------Imagem 05 -------------------------
    #---------------------------------------------------
    imagem05("images/image_05.png", 'image_05_fourier.png')
    
    #*********Filtro Gaussiano**********
    filtro2(path="images/image_05.png", function=fGauss, media=0, desvio=10, dim=9, repeticoes=1, nome="image_05_gauss9x9.png")
    filtro2(path="images/image_05.png", function=fGauss, media=0, desvio=10, dim=7, repeticoes=1, nome="image_05_gauss7x7.png")
    filtro2(path="images/image_05.png", function=fGauss, media=0, desvio=10, dim=5, repeticoes=1, nome="image_05_gauss5x5.png")
    filtro2(path="images/image_05.png", function=fGauss, media=0, desvio=10, dim=3, repeticoes=1, nome="image_05_gauss3x3.png")
    
    hist.carrega("images/image_05.png", "image_05_gauss9x9.png", "image_05", "image_05_gauss9x9.png")
    hist.carrega("images/image_05.png", "image_05_gauss7x7.png", "image_05", "image_05_gauss7x7.png")
    hist.carrega("images/image_05.png", "image_05_gauss5x5.png", "image_05", "image_05_gauss5x5.png")
    hist.carrega("images/image_05.png", "image_05_gauss3x3.png", "image_05", "image_05_gauss3x3.png")
    
    image5 = np.array(pil.open("images/image_05.png"))
    image5_9x9 = np.array(pil.open("image_05_gauss9x9.png"))
    image5_7x7 = np.array(pil.open("image_05_gauss7x7.png"))
    image5_5x5 = np.array(pil.open("image_05_gauss5x5.png"))
    image5_3x3 = np.array(pil.open("image_05_gauss3x3.png"))
    
    hist.histograma([image5 - image5_9x9], 1, ["Ruido - image_05_gauss9x9"], "Ruido - image_05_gauss9x9")
    hist.histograma([image5 - image5_7x7], 1, ["Ruido - image_05_gauss7x7"], "Ruido - image_05_gauss7x7")
    hist.histograma([image5 - image5_5x5], 1, ["Ruido - image_05_gauss5x5"], "Ruido - image_05_gauss5x5")
    hist.histograma([image5 - image5_3x3], 1, ["Ruido - image_05_gauss3x3"], "Ruido - image_05_gauss3x3")

    #*********Filtro Média Harmônica**********
    statistic_filter("images/image_05.png", 3, mediaHarmonica, 'image05_mediaharmonica_3x3.png')
    statistic_filter("images/image_05.png", 5, mediaHarmonica, 'image05_mediaharmonica_5x5.png')
    statistic_filter("images/image_05.png", 7, mediaHarmonica, 'image05_mediaharmonica_7x7.png')

    hist.carrega("images/image_05.png", "image05_mediaharmonica_3x3.png", "image_05", "image05_mediaharmonica_3x3")
    hist.carrega("images/image_05.png", "image05_mediaharmonica_5x5.png", "image_05", "image05_mediaharmonica_5x5")
    hist.carrega("images/image_05.png", "image05_mediaharmonica_7x7.png", "image_05", "image05_mediaharmonica_7x7")

    image05_mediaharmonica_3x3 = np.array(pil.open("image05_mediaharmonica_3x3.png"))
    image05_mediaharmonica_5x5 = np.array(pil.open("image05_mediaharmonica_5x5.png"))
    image05_mediaharmonica_7x7 = np.array(pil.open("image05_mediaharmonica_7x7.png"))
    
    hist.histograma([image5 - image05_mediaharmonica_3x3], 1, ["Ruido - image05_mediaharmonica_3x3"], "Ruido - image05_mediaharmonica_3x3")
    hist.histograma([image5 - image05_mediaharmonica_5x5], 1, ["Ruido - image05_mediaharmonica_5x5"], "Ruido - image05_mediaharmonica_5x5")
    hist.histograma([image5 - image05_mediaharmonica_7x7], 1, ["Ruido - image05_mediaharmonica_7x7"], "Ruido - image05_mediaharmonica_7x7")

    #*********Filtro Adaptativo**********
    adaptative_filter("images/image_05.png", 3, 100, 'image05_adaptative_3x3.png')
    adaptative_filter("images/image_05.png", 5, 100, 'image05_adaptative_5x5.png')
    adaptative_filter("images/image_05.png", 7, 100, 'image05_adaptative_7x7.png')

    hist.carrega("images/image_05.png", "image05_adaptative_3x3.png", "image_05", "image05_adaptative_3x3")
    hist.carrega("images/image_05.png", "image05_adaptative_5x5.png", "image_05", "image05_adaptative_5x5")
    hist.carrega("images/image_05.png", "image05_adaptative_7x7.png", "image_05", "image05_adaptative_7x7")

    image05_adaptative_3x3 = np.array(pil.open("image05_adaptative_3x3.png"))
    image05_adaptative_5x5 = np.array(pil.open("image05_adaptative_5x5.png"))
    image05_adaptative_7x7 = np.array(pil.open("image05_adaptative_7x7.png"))

    hist.histograma([image5 - image05_adaptative_3x3], 1, ["Ruido - image05_adaptative_3x3"], "Ruido - image05_adaptative_3x3")
    hist.histograma([image5 - image05_adaptative_5x5], 1, ["Ruido - image05_adaptative_5x5"], "Ruido - image05_adaptative_5x5")
    hist.histograma([image5 - image05_adaptative_7x7], 1, ["Ruido - image05_adaptative_7x7"], "Ruido - image05_adaptative_7x7")

          
    realceGama("image_05_gauss3x3.png", 140, 0.62, "image_05_gauss3x3")
    realceHistEq("image_05_gauss3x3.png", 255, "image_05_gauss3x3")
