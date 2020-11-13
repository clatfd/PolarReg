import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#augmentation by shifting/rotating polar patches
def polarrot(polararr,offset):
    patchheight = polararr.shape[0]
    polarpatchoff = np.zeros(polararr.shape)
    polarpatchoff[offset:] = polararr[:patchheight - offset]
    polarpatchoff[:offset] = polararr[patchheight - offset:]
    return polarpatchoff

#augmentation by shifting/rotating polar patches in batch
def batch_polar_rot(polararr,offset):
    patchheight = polararr.shape[1]
    polarpatchoff = np.zeros(polararr.shape)
    try:
        polarpatchoff[:,offset:] = polararr[:,:patchheight - offset]
    except:
        print(offset,polararr.shape,polarpatchoff.shape)
        raise TypeError
    polarpatchoff[:,:offset] = polararr[:,patchheight - offset:]
    return polarpatchoff

#from polar contour to cart contour
def toctbd(polarbd,ctx,cty):
    # height, in/out bd, x/y pos
    contour1 = []
    contour2 = []
    for offi in range(polarbd.shape[0]):
        ctheta = 360 / polarbd.shape[0] * offi
        crho = polarbd[offi, 0]
        cx = crho * math.cos(ctheta / 180 * np.pi)
        cy = crho * math.sin(ctheta / 180 * np.pi)
        contour1.append([ctx + cx, cty + cy])
        crho = polarbd[offi, 1]
        cx = crho * math.cos(ctheta / 180 * np.pi)
        cy = crho * math.sin(ctheta / 180 * np.pi)
        contour2.append([ctx + cx, cty + cy])
    contourout = np.array(contour2)
    contourin = np.array(contour1)
    return contourin, contourout

def polarpredimg(polarpatch,config,polarmodel):
    patchheight = config['patchheight']
    height = config['height']
    patchwidth = config['width']
    width = config['width']
    depth = config['depth']
    channel = config['channel']

    imgpad = patchheight
    test_patch_batch = np.zeros((height + imgpad, patchheight, patchwidth, depth, channel))

    # vertical extend both direction
    pradout = np.zeros((height + 2 * imgpad, patchwidth, depth, channel))
    pradout[:imgpad] = polarpatch[-imgpad:]
    pradout[imgpad:-imgpad] = polarpatch
    pradout[-imgpad:] = polarpatch[:imgpad]
    # result patch
    pred_batch_pad = np.zeros((height + 2 * imgpad, 2))
    # pts = [[] for i in range(height+2*imgpad)]
    pred_batch_img_pad = np.zeros((height + 2 * imgpad, width, depth, 2))
    pred_batch_ct = np.zeros((height + 2 * imgpad, 1))
    # collect patches
    for offi in range(height + imgpad):
        test_patch_batch[offi] = pradout[offi:offi + patchheight]
        # print(pradout[offi,0,1,0],pradout[offi+patchheight-1,0,1,0])

    predimg_batch, pred_batch = polarmodel.predict(test_patch_batch)

    for offi in range(height + imgpad):
        pred_batch_pad[offi:offi + patchheight] += pred_batch[offi]
        pred_batch_img_pad[offi:offi + patchheight] += predimg_batch[offi]
        pred_batch_ct[offi:offi + patchheight] += 1
        # for slicei in range(80):
        #    pts[offi+slicei].append(pred_batch[offi][slicei,0])

    pred_img_c = np.zeros((height, width, depth, 2))
    pred_contour_c = np.zeros((height, 2))
    for offi in range(height):
        pred_img_c[offi] = pred_batch_img_pad[offi + imgpad] / pred_batch_ct[offi + imgpad]
        pred_contour_c[offi] = pred_batch_pad[offi + imgpad] / pred_batch_ct[offi + imgpad] * width

    # print(pts[imgpad],np.mean(pts[imgpad]))
    # print(pts[height-1+imgpad],np.mean(pts[height-1+imgpad]))
    return pred_img_c, pred_contour_c

from PolarVW.grad import get_grad_img
import copy
# rotate patch and predict polar cont along with consistency in rotated patches
def polar_pred_cont_cst(polarpatch,config,polarmodel,gap=10,bioutput=False,usegrad=False):
    DEBUG = 0
    patchheight = config['patchheight']
    width = config['width']
    depth = config['depth']
    channel = config['channel']
    totalheight = config['height']

    # result patch
    # pred_batch_img = np.zeros((totalheight,width,depth,2))
    pred_batch_num = np.zeros((totalheight))
    pred_batch_ctlist = [[] for i in range(totalheight)]

    polarpatchpad = np.zeros((totalheight + patchheight, width, depth, channel))
    polarpatchpad[:totalheight] = polarpatch
    polarpatchpad[totalheight:] = polarpatch[:patchheight]

    test_patch_batch = []

    # collect patches
    for offi in range(0, totalheight, gap):
        polarimgrot = polarpatchpad[offi:offi + patchheight]
        test_patch_batch.append(polarimgrot)

    if bioutput:
        _, predct_batch = polarmodel.predict(np.array(test_patch_batch))
    else:
        predct_batch = polarmodel.predict(np.array(test_patch_batch))

    if usegrad == True:
        polargrad  = get_grad_img(polarpatch[:,:,1,0])

        for bi in range(len(predct_batch)):
            offi = bi * gap
            for slicei in range(patchheight):
                weight = polargrad[(offi + slicei) % totalheight,int(round(predct_batch[bi][slicei][0]*width))]
                #print(bi,(offi + slicei) % totalheight,int(round(predct_batch[bi][slicei][0]*width)),weight)
                pred_batch_ctlist[(offi + slicei) % totalheight].append(predct_batch[bi][slicei]*weight)
                pred_batch_num[(offi + slicei) % totalheight] += weight
    else:
        for bi in range(len(predct_batch)):
            offi = bi * gap
            # print(offi%totalheight,(offi+patchheight-1)%totalheight)
            # pred_batch_img[totalheight-offi:] += predimg_batch[bi][:offi]
            # pred_batch_num[totalheight-offi:] += 1
            # pred_batch_img[:totalheight-offi] += predimg_batch[bi][offi:]
            # pred_batch_num[:totalheight-offi] += 1

            for slicei in range(patchheight):
                pred_batch_ctlist[(offi + slicei) % totalheight].append(predct_batch[bi][slicei])

    # pred_img = np.zeros((totalheight,width,depth,2))
    pred_contour = np.zeros((totalheight, 2))
    pred_contour_sd = np.zeros((totalheight, 2))

    if usegrad == True:
        polargraddsp = copy.copy(polargrad)
        polarpatchdsp = polarpatch[:, :, 1, 0]

    for offi in range(totalheight):
        # pred_img[offi] = pred_batch_img[offi]/pred_batch_num[offi]
        if usegrad == True:
            #print(offi,np.sum(pred_batch_ctlist[offi], axis=0), pred_batch_num[offi])
            pred_contour[offi] = np.sum(pred_batch_ctlist[offi], axis=0)/pred_batch_num[offi] * width
            polargraddsp[offi,int(round(pred_contour[offi][0]))] = np.max(polargraddsp)
            polarpatchdsp[offi,int(round(pred_contour[offi][0]))] = np.max(polarpatchdsp)
        else:
            pred_contour[offi] = np.mean(pred_batch_ctlist[offi], axis=0) * width
        # std dev for uniform distribution in range 0-1 is (1-0)/sqrt(12)
        pred_contour_sd[offi] = np.std(pred_batch_ctlist[offi], axis=0) * np.sqrt(12)

    if DEBUG==1 and usegrad == True:
        plt.imshow(polargraddsp)
        plt.show()
        plt.imshow(polarpatchdsp)
        plt.show()

    # print('batch',len(predimg_batch))
    # print([(i,pred_batch_num[i]) for i in range(len(pred_batch_num))])
    # pyplot.plot([len(i) for i in pred_batch_ctlist])
    # pyplot.show()
    # print([pred_batch_ctlist[0][i][0] for i in range(len(pred_batch_ctlist[0]))])
    # print([pred_batch_ctlist[-1][i][0] for i in range(len(pred_batch_ctlist[-1]))])
    return pred_contour, pred_contour_sd


def plotct(sz, contourin, contourout):
    imgmask = np.zeros((sz, sz), dtype=np.uint8)
    intcont = []
    for conti in range(len(contourout)):
        intcont.append([int(round(contourout[conti][0])), int(round(contourout[conti][1]))])
    intcont = np.array(intcont)
    cv2.fillPoly(imgmask, pts=[intcont], color=(1, 1, 1))
    if contourin is not None:
        intcont = []
        for conti in range(len(contourin)):
            intcont.append([int(round(contourin[conti][0])), int(round(contourin[conti][1]))])
        intcont = np.array(intcont)
        cv2.fillPoly(imgmask, pts=[intcont], color=(0, 0, 0))
    # pyplot.imshow(imgmask)
    # pyplot.show()
    return imgmask

def plotpolar(rho, polarbd):
    imgmask = np.zeros((len(polarbd), rho), dtype=np.uint8)
    for conti in range(len(polarbd)):
        imgmask[conti, int(round(polarbd[conti][0] + 1)):int(round(polarbd[conti][1] + 1))] = 1
    return imgmask

def draw_com_vw(com_vw_name,cart_vw_label,cart_vw_seg,cartpatch):
    comimg = np.repeat((cartpatch[:,:,None]*255).astype(np.uint8),3,axis=2)
    comimg[:, :, 0] = cart_vw_label * 255
    comimg[:, :, 1] = cart_vw_seg * 255
    cv2.imwrite(com_vw_name,comimg)

from PolarVW.eval import diffmap,DSC
def polar_plot(polar_plot_name,cart_patch,cart_label,polarbd,cart_seg,
               polar_padisttch,polar_label,cart_unet,cart_mask):
    fz = 20
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(2, 5, 1)
    plt.title('Cartesian Patch', fontsize=fz)
    plt.imshow(cart_patch, cmap='gray')
    plt.subplot(2, 5, 2)
    plt.title('Cartesian Label', fontsize=fz)
    plt.imshow(cart_label, cmap='gray')
    plt.subplot(2, 5, 3)
    plt.title('Prediction \nDSC:%.3f' % (DSC(cart_seg, cart_label)), fontsize=fz)
    plt.imshow(diffmap(cart_seg, cart_label), cmap='gray')

    if cart_unet is not None:
        plt.subplot(2, 5, 4)
        plt.title('U-Net Prediction \nDSC:%.3f' % (DSC(cart_unet, cart_label)), fontsize=fz)
        plt.imshow(diffmap(cart_unet, cart_label), cmap='gray')
    if cart_mask is not None:
        plt.subplot(2, 5, 5)
        plt.title('Mask-RCNN Prediction \nDSC:%.3f' % (DSC(cart_mask, cart_label)), fontsize=fz)
        plt.imshow(diffmap(cart_mask, cart_label), cmap='gray')

    plt.subplot(2, 5, 6)
    plt.title('Polar Patch', fontsize=fz)
    plt.imshow(polar_patch, cmap='gray')
    plt.subplot(2, 5, 7)
    plt.title('Polar Label', fontsize=fz)
    plt.imshow(polar_label, cmap='gray')
    plt.subplot(2, 5, 8)
    plt.title('Polar Prediction', fontsize=fz)
    plt.xlim([0, 256])
    plt.ylim([polarbd.shape[0], 0])
    plt.plot(polarbd[::4, 0], np.arange(0,polarbd.shape[0],4), 'o', markersize=2, label='Lumen')
    plt.plot(polarbd[::4, 1], np.arange(0,polarbd.shape[0],4), 'o', markersize=2,  label='Wall')
    plt.legend()

    # fig.tight_layout()
    if polar_plot_name is not None:
        plt.savefig(polar_plot_name)
    else:
        plt.show()
    plt.close()

def tocordpolar(cartcord, octx, octy, rsamples=256, thsamples=256):
    thetas = []
    rhostheta = {}

    for pti in range(len(cartcord)):
        wx = cartcord[pti][0] - octx
        hy = cartcord[pti][1] - octy
        crho = np.sqrt(hy * hy + wx * wx) / rsamples
        cth = math.atan2(hy, wx) / np.pi * 180
        if cth < 0:
            cth = (360 + cth)
        else:
            cth = (cth)
        thetas.append(cth)
        rhostheta[cth] = crho

    thetas = np.sort(thetas)

    # add last first degree outside 0-360 to allow full range of interp from 0-360
    thetas_add = [np.max(thetas)-360]
    rhostheta[np.max(thetas)-360] = rhostheta[np.max(thetas)]
    thetas_add.extend(copy.copy(thetas))
    thetas_add.append(np.min(thetas)+360)
    rhostheta[np.min(thetas) + 360] = rhostheta[np.min(thetas)]
    rhos_add = [rhostheta[t] for t in thetas_add]

    f1 = interp1d(thetas_add, rhos_add)
    step = 360 / thsamples
    interprho = f1(np.arange(thsamples) * step)
    return interprho


def exportpolarcontour(coutourname, polarcontstack):
    with open(coutourname, 'w') as contourfile:
        for contindx in range(len(polarcontstack[1])):
            contourfile.write("%.7f %.7f\n" % (polarcontstack[0][contindx], polarcontstack[1][contindx]))


#rotation pred polar ct from rot 90 cart patch
def pred_polar_ct_rot(cnn,cartstackrz,cfg):
    FIG = 0
    rots = 4
    cartstackrzbatch = np.zeros((max(rots,cfg['G']),cartstackrz.shape[0],cartstackrz.shape[1],cartstackrz.shape[2]))
    cartstackrzbatch[0] = cartstackrz
    for roti in range(1,rots):
        cartstackrz = np.rot90(cartstackrz)
        cartstackrzbatch[roti] = cartstackrz
    polarcts = cnn.predict(cartstackrzbatch[...,None])*256
    polar_bd_array = np.zeros((cfg['patchheight'],2,rots))
    for roti in range(rots):
        polarrot = polarcts[roti]
        polarrot_o = np.concatenate([polarrot[-256//rots*roti:],polarrot[:-256//rots*roti]])
        polar_bd_array[:,:,roti] = polarrot_o
        contourin,contourout = toctbd(polarcts[roti],cfg['height'],cfg['height'])
        cart_vw_seg = plotct(512,contourin,contourout)
        if FIG:
            plt.subplot(1,2,1)
            plt.imshow(cartstackrzbatch[roti])
            plt.subplot(1,2,2)
            plt.imshow(cart_vw_seg)
            plt.suptitle(roti)
            plt.show()
    return polar_bd_array

def batch_cart_rot(xarray,yarray,rotsarr):
    xarray_off = np.zeros(xarray.shape)
    yarray_off = np.zeros(yarray.shape)
    for i in range(len(rotsarr)):
        roti = rotsarr[i]
        xrot = xarray[i]
        yrot = yarray[i]
        if roti!=0:
            xrot = np.rot90(xrot,roti)
            yrot = np.concatenate([yrot[64*roti:],yrot[:64*roti]])
        xarray_off[i] = xrot
        yarray_off[i] = yrot
    return xarray_off, yarray_off

def resamplecont(cont,resamplesize):
    f1 = interp1d(np.arange(len(cont)),cont)
    interprho = f1(np.arange(resamplesize)*len(cont)/resamplesize)
    return interprho

def resampleconts(conts,resamplesize):
    resampleconts = np.zeros((resamplesize,2))
    resampleconts[:,0] = resamplecont(conts[:,0], resamplesize)
    resampleconts[:,1] = resamplecont(conts[:,1], resamplesize)
    return resampleconts
