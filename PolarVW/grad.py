from scipy import ndimage
from scipy.ndimage import gaussian_filter
import numpy as np
from PolarVW.UTL import croppatch
import copy
import matplotlib.pyplot as plt


def get_grad_img(polar_patch):
    OFFX = 2
    polar_patch_gaussian = gaussian_filter(polar_patch, sigma=5)
    sy = ndimage.sobel(polar_patch_gaussian, axis=0, mode='constant')
    sx = ndimage.sobel(polar_patch_gaussian, axis=1, mode='constant')
    polar_grad = np.hypot(sx, sy)
    polar_grad = croppatch(croppatch(polar_grad, 128, 128, 127, 127), 127, 127-OFFX, 128, 128)
    polar_grad[0] = polar_grad[1]
    polar_grad[-1] = polar_grad[-2]
    gradimg = polar_grad #/ np.max(polar_grad)
    return gradimg


def contgrad(polar_patch,polar_cont):
    DEBUG = 0
    polar_grad = get_grad_img(polar_patch)
    if DEBUG==1:
        polar_grad_dsp = copy.copy(polar_grad)
        for i in range(polar_cont.shape[0]):
            polar_grad_dsp[i, int(round(polar_cont[i, 0] * 256))] = 1
        #plt.imshow(polar_grad_dsp)
        #plt.show()
    grad = [np.mean(polar_grad[i, int(round(polar_cont[i, 0] * 256)) - 1 : int(round(polar_cont[i, 0] * 256)) + 2])
            for i in range(polar_cont.shape[0])]
    meangrad = np.mean(grad)
    if np.isnan(meangrad):
        print('nan occur',grad)
        meangrad = 0.1
    return meangrad
