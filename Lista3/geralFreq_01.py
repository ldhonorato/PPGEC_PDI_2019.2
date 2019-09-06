import pdi_atividade03 as filtrosFreq
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

#===================================================
#------------Tratamento para Suavizar 01------------
#===================================================
imgPath = "images/geral_01.png"
originalImg = np.array(pil.open(imgPath)).astype(np.int16)


for d0 in (25, 50, 100, 150, 180, 200):
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.LPF, "lpf_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.GLPF, "GLPF_"+str(d0))
    filtrosFreq.aplica(imgPath, d0, filtrosFreq.BLPF, "BLPF_n2_"+str(d0), n=2)

filtrosFreq.aplica("geral_01_BLPF_n2_200.png", 200, filtrosFreq.GLPF, "GLPF_200")
filtrosFreq.aplica("geral_01_BLPF_n2_200.png", 300, filtrosFreq.GLPF, "GLPF_300")
filtrosFreq.aplica("geral_01_BLPF_n2_200.png", 300, filtrosFreq.LPF, "LPF_300")
filtrosFreq.aplica("geral_01_lpf_180.png", 100, filtrosFreq.GLPF, "GLPF_100")
