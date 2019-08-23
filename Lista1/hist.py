# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:00:16 2019

@author: Agostinho
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil

def show_images(images, cols = 1, titles = None, index = "", pasta = ""):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.hist(image.flatten(), 256, [0, 256])
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 3)
    fig.savefig(pasta + "hist_"+ str(index) + ".png")

def carrega(im1, im2, nome1, nome2, pasta):
    img = pil.open(im1)
    img = np.array(img)
    
    img2 = pil.open(im2)
    img2 = np.array(img2)
    
    show_images([img, img2], 1, ["Original - " + nome1, "Resultado - " + nome2], nome2, pasta)