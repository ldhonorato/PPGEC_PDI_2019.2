import pdi_atividade03 as filtrosFreq
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

#===================================================
#------------Tratamento para Suavizar 01------------
#===================================================
imgPath = "images/suavizar_01.png"
originalImg = np.array(pil.open(imgPath)).astype(np.int16)


filtrosFreq.aplica(imgPath, 250, filtrosFreq.LPF, "lpf_250")
filtrosFreq.aplica(imgPath, 250, filtrosFreq.GLPF, "GLPF_250")
filtrosFreq.aplica(imgPath, 150, filtrosFreq.GLPF, "GLPF_150")
filtrosFreq.aplica(imgPath, 250, filtrosFreq.BLPF, "BLPF_250_n2", n=2)
filtrosFreq.aplica(imgPath, 450, filtrosFreq.LPF, "lpf_450")
filtrosFreq.aplica(imgPath, 450, filtrosFreq.GLPF, "GLPF_450")
filtrosFreq.aplica(imgPath, 450, filtrosFreq.BLPF, "BLPF_450_n2", n=2)
filtrosFreq.aplica("suavizar_01_BLPF_250_n2.png", 200, filtrosFreq.GLPF, "GLPF_200")
filtrosFreq.aplica("suavizar_01_BLPF_250_n2.png", 300, filtrosFreq.GLPF, "GLPF_300")
filtrosFreq.aplica("suavizar_01_BLPF_250_n2.png", 300, filtrosFreq.LPF, "LPF_300")
