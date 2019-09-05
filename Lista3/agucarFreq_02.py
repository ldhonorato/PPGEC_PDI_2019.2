import pdi_atividade03 as filtrosFreq
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math

#===================================================
#------------Tratamento para Agucar 02--------------
#===================================================
imgPath = "images/agucar_02.png"

filtrosFreq.aplica(imgPath, 10, filtrosFreq.HPF, "hpf_10")
filtrosFreq.aplica(imgPath, 30, filtrosFreq.HPF, "hpf_30")

filtrosFreq.aplica(imgPath, 10, filtrosFreq.GHPF, "ghpf_10")
filtrosFreq.aplica(imgPath, 30, filtrosFreq.GHPF, "ghpf_30")

filtrosFreq.aplica(imgPath, 10, filtrosFreq.BHPF, "bhpf_10")
filtrosFreq.aplica(imgPath, 30, filtrosFreq.BHPF, "bhpf_30")



img_orig = np.array(pil.open('images/agucar_02.png'))
img_origNormalizada = filtrosFreq.normalizar(img_orig)
#-------------------------------
#----Agucamento Laplaciano------
#-------------------------------
img_lapla = filtrosFreq.aplica(imgPath, 0, filtrosFreq.LAPLA, "lapla")
for c in (1, 5, 20, 30, 40, 50):
    g_lapla = img_origNormalizada - c*img_lapla
    g_lapla = (g_lapla/np.max(g_lapla))*255
    g_lapla = pil.fromarray(g_lapla)
    g_lapla = g_lapla.convert("L")
    g_lapla.save("agucar_02_agucamentoLaplace" + str(c) + ".png")

    filtrosFreq.aplica("agucar_02_agucamentoLaplace" + str(c) + ".png", 250, filtrosFreq.GLPF, "GLPF250")

#===============================
#------Mascara de Nitidez-------
#===============================
filtrosFreq.aplica(imgPath, 100, filtrosFreq.GLPF, "GLPF_100")
filtrosFreq.aplica(imgPath, 50, filtrosFreq.GLPF, "GLPF_50")

img_glpf_50 = np.array(pil.open('agucar_02_GLPF_50.png'))
img_glpf_100 = np.array(pil.open('agucar_02_GLPF_100.png'))

img_glpf_50 = filtrosFreq.normalizar(img_glpf_50)
img_glpf_100 = filtrosFreq.normalizar(img_glpf_100)

for k in (1, 5, 10):
    mask_50 = img_origNormalizada - img_glpf_50
    mask_100 = img_origNormalizada - img_glpf_100

    g_50 = img_origNormalizada + k*mask_50
    g_100 = img_origNormalizada + k*mask_100

    g_50 = (g_50 / np.max(g_50))*255
    g_100 = (g_100 / np.max(g_100))*255\

    filtrosFreq.saveImgFromArray(mask_50, "agucar_02_mask_GLPF_50_k" + str(k) + ".png")
    filtrosFreq.saveImgFromArray(mask_100, "agucar_02_mask_GLPF_100_k" + str(k) + ".png")
    filtrosFreq.saveImgFromArray(g_50, "agucar_02_Final_mask_GLPF_50_k" + str(k) + ".png")
    filtrosFreq.saveImgFromArray(g_100, "agucar_02_Final_mask_GLPF_100_k" + str(k) + ".png")

    filtrosFreq.realceHistEq("agucar_02_Final_mask_GLPF_50_k" + str(k) + ".png", 180, "agucar_02_Final_mask_GLPF_50_k" + str(k))
    filtrosFreq.realceHistEq("agucar_02_Final_mask_GLPF_100_k" + str(k) + ".png", 180, "agucar_02_Final_mask_GLPF_100_k" + str(k))


#===================================================
#------Filtragem de ênfase de alta frequencia-------
#===================================================
#filtroAltaEnfaseFrequencia(imgPath, 30, GHPF, "ENFASE_ALTA_FREQ", k1=0.5, k2=0.75)
