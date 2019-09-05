import pdi_atividade03 as filtrosFreq
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

#===================================================
#------------Tratamento para Suavizar 02------------
#===================================================
imgPath = "images/suavizar_02.png"
originalImg = np.array(pil.open(imgPath)).astype(np.int16)

#Filtro MARRETA
marretaImg = np.copy(originalImg)
for i in range(40, originalImg.shape[0], 40):
    marretaImg[i,:] = (marretaImg[i-1,:] + marretaImg[i+1,:])/2

for i in range(40, originalImg.shape[1], 40):
    marretaImg[:,i] = (marretaImg[:,i-1] + marretaImg[:,i+1])/2

filtrosFreq.saveImgFromArray(marretaImg, "suavizar02_filtroMarreta.png")

originalImg_pad = filtrosFreq.padding2(originalImg)
originalImg_pad = originalImg_pad/255.
imgfft = np.fft.fft2(originalImg_pad)
imgfftshift = np.fft.fftshift(imgfft)
spectro = 20*np.log(np.abs(imgfftshift))

marretaImg = np.array(pil.open("suavizar02_filtroMarreta.png")).astype(np.int16)
marretaImg_pad = filtrosFreq.padding2(marretaImg)
marretaImg_pad = marretaImg_pad/255.
marreta_imgfft = np.fft.fft2(marretaImg_pad)
marreta_imgfftshift = np.fft.fftshift(marreta_imgfft)
marreta_spectro = 20*np.log(np.abs(marreta_imgfftshift))

fig = plt.figure()
fig.suptitle('Analise do espectro')
fig.add_subplot(1, 3, 1)
plt.imshow(spectro, cmap='gray')
plt.title("Espectro da imagem original")
fig.add_subplot(1, 3, 2)
plt.imshow(marreta_spectro, cmap='gray')
plt.title("Espectro da imagem marreta")
fig.add_subplot(1, 3, 3)
plt.imshow(marreta_spectro-spectro, cmap='gray')
plt.title("Diferença entre os espectros")
plt.show()

for d0 in (25, 50, 100, 150, 200):
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.LPF, "lpf_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.GLPF, "GLPF_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.BLPF, "BLPF_n2_"+str(d0), n=2)

for d0 in (25, 50, 100, 150, 200):
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.HPF, "hpf_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.GHPF, "GHPF_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.BHPF, "BHPF_n2_"+str(d0), n=2)

bhpfImage = filtrosFreq.aplica(imgPath, 250, filtrosFreq.BHPF, "BHPF_n2_250", n=2)



