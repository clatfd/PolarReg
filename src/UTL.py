import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
import cv2
from scipy.interpolate import interp1d
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import matplotlib.pyplot as pyplot
import datetime

def fitimg(img,targetshape):
    if(len(targetshape)==2):
        if img.shape[0]>targetshape[0]:
            offshi = img.shape[0]-targetshape[0]
            img = img[offshi//2:img.shape[0]-(offshi-offshi//2),:]
        else:
            offshi = targetshape[0]-img.shape[0]
            imgz = np.zeros((targetshape[0],img.shape[1]))
            imgz[offshi//2:targetshape[0]-(offshi-offshi//2),:] = img
            img = imgz
        if img.shape[1]>targetshape[1]:
            offshi = img.shape[1]-targetshape[1]
            img = img[:,offshi//2:img.shape[1]-(offshi-offshi//2)]
        else:
            offshi = targetshape[1]-img.shape[1]
            imgz = np.zeros((img.shape[0],targetshape[1]))
            imgz[:,offshi//2:targetshape[1]-(offshi-offshi//2)] = img
            img = imgz
    elif(len(targetshape)==3):
        if img.shape[0]>targetshape[0]:
            offshi = img.shape[0]-targetshape[0]
            img = img[offshi//2:img.shape[0]-(offshi-offshi//2),:,:]
        else:
            offshi = targetshape[0]-img.shape[0]
            imgz = np.zeros((targetshape[0],img.shape[1],img.shape[2]))
            imgz[offshi//2:targetshape[0]-(offshi-offshi//2),:,:] = img
            img = imgz
        if img.shape[1]>targetshape[1]:
            offshi = img.shape[1]-targetshape[1]
            img = img[:,offshi//2:img.shape[1]-(offshi-offshi//2),:]
        else:
            offshi = targetshape[1]-img.shape[1]
            imgz = np.zeros((img.shape[0],targetshape[1],img.shape[2]))
            imgz[:,offshi//2:targetshape[1]-(offshi-offshi//2),:] = img
            img = imgz
        if img.shape[2]>targetshape[2]:
            offshi = img.shape[2]-targetshape[2]
            img = img[:,:,offshi//2:img.shape[2]-(offshi-offshi//2)]
        else:
            offshi = targetshape[2]-img.shape[2]
            imgz = np.zeros((img.shape[0],img.shape[1],targetshape[2]))
            imgz[:,:,offshi//2:targetshape[2]-(offshi-offshi//2)] = img
            img = imgz
    else:
        print('len not 2/3')
    return img


def predimg(cnn,patch,channel,batch_size=2048):
    #predict img with eight half sized patches
    #validated all 1
    half_stride=8
    width=16
    height=16
    depth=16

    c_patch=np.zeros((batch_size,depth,width,height,channel))
    patchpos=np.zeros((batch_size,3))
    patchloc=np.zeros((batch_size,3),dtype='int32')
    bi=0

    print('original size:',patch.shape)
    trim_patch_shape=((patch.shape[0]//depth)*depth,(patch.shape[1]//width)*width,(patch.shape[2]//height)*height)
    patch_trim=patch[:(patch.shape[0]//depth)*depth,:(patch.shape[1]//width)*width,:(patch.shape[2]//height)*height]
    trim_patch_shape = patch_trim.shape
    print('Trim to:',trim_patch_shape)
    predict_img=np.zeros((patch.shape[0],patch.shape[1],patch.shape[2]))

    for i in range(trim_patch_shape[0]//half_stride-1):
        print("\rdepth", i, '/', (trim_patch_shape[0])//half_stride-1, end="") 
        for j in range(trim_patch_shape[1]//half_stride-1):
            for k in range(trim_patch_shape[2]//half_stride-1):
                img= patch_trim[i*half_stride:i*half_stride+depth,
                                j*half_stride:j*half_stride+width,
                                k*half_stride:k*half_stride+height]
                
                patchloc[bi,:]=(i,j,k)
                if channel==1:
                    c_patch[bi,:,:,:,0]=img
                else:
                    c_patch[bi,:,:,:]=img
                #patchpos[bi,:]=[(i*half_stride+depth//2)/patch.shape[0],(j*half_stride+width//2)/patch.shape[1],(k*half_stride+height//2)/patch.shape[2]]
                patchpos[bi,:]=[(i*half_stride+depth//2)/trim_patch_shape[0],
                                (j*half_stride+width//2)/trim_patch_shape[1],
                                (k*half_stride+height//2)/trim_patch_shape[2]]
                bi=bi+1
                if bi==batch_size or \
                        i==trim_patch_shape[0]//half_stride-2 \
                    and j==trim_patch_shape[1]//half_stride-2 \
                    and k==trim_patch_shape[2]//half_stride-2:
                    test_out = cnn.predict([c_patch,patchpos])
                    #test_out = np.ones((bi,depth,width,height,1))
                    for bbi in range(bi):
                        cdi = patchloc[bbi,0]
                        cwi = patchloc[bbi,1]
                        chi = patchloc[bbi,2]

                        predict_img[cdi * half_stride : cdi*half_stride+depth,
                                   cwi * half_stride : cwi*half_stride+width,
                                   chi *half_stride: chi*half_stride+height]+=test_out[bbi].reshape(depth,width,height)
                    #reset
                    bi=0
    pend = (patch.shape[0]//depth)*depth
    predict_img[half_stride:pend-half_stride,
                half_stride:trim_patch_shape[1]-half_stride,
                half_stride:trim_patch_shape[2]-half_stride] = predict_img[half_stride:trim_patch_shape[0]-half_stride,half_stride:trim_patch_shape[1]-half_stride,half_stride:trim_patch_shape[2]-half_stride]/8.
    predict_img[0:half_stride,half_stride:-half_stride,half_stride:-half_stride]/=4
    predict_img[half_stride:pend-half_stride,0:half_stride,half_stride:-half_stride]/=4
    predict_img[half_stride:pend-half_stride,half_stride:-half_stride,0:half_stride]/=4
    predict_img[pend-half_stride:pend,half_stride:-half_stride,half_stride:-half_stride]/=4
    predict_img[half_stride:pend-half_stride,-half_stride:,half_stride:-half_stride]/=4
    predict_img[half_stride:pend-half_stride,half_stride:-half_stride,-half_stride:]/=4
    
    predict_img[:half_stride,half_stride:-half_stride,:half_stride]/=2
    predict_img[:half_stride,half_stride:-half_stride,-half_stride:]/=2
    predict_img[:half_stride,:half_stride,half_stride:-half_stride]/=2
    predict_img[:half_stride,-half_stride:,half_stride:-half_stride:]/=2
    
    predict_img[pend-half_stride:pend,half_stride:-half_stride,:half_stride]/=2
    predict_img[pend-half_stride:pend,half_stride:-half_stride,-half_stride:]/=2
    predict_img[pend-half_stride:pend,:half_stride,half_stride:-half_stride]/=2
    predict_img[pend-half_stride:pend,-half_stride:,half_stride:-half_stride:]/=2
    
    predict_img[half_stride:pend-half_stride,:half_stride,:half_stride]/=2
    predict_img[half_stride:pend-half_stride,:half_stride,-half_stride:]/=2
    predict_img[half_stride:pend-half_stride,-half_stride:,:half_stride]/=2
    predict_img[half_stride:pend-half_stride,-half_stride:,-half_stride:]/=2
    
    if (patch.shape[0]//depth)*depth<patch.shape[0]:
        #last batch of slice
        print('\nPredicting last slices in depth')
        pad_patch_shape=(depth,trim_patch_shape[1],trim_patch_shape[2],3)
        patch_pad=patch[patch.shape[0]-depth:patch.shape[0],:trim_patch_shape[1],:trim_patch_shape[2]]

        for j in range(pad_patch_shape[1]//half_stride-1):
            for k in range(pad_patch_shape[2]//half_stride-1):
                img= patch_pad[:,j*half_stride:j*half_stride+width,k*half_stride:k*half_stride+height]           

                patchloc[bi,:]=(patch.shape[0],j,k)
                if channel==1:
                    c_patch[bi,:,:,:,0]=img
                else:
                    c_patch[bi,:,:,:]=img
                patchpos[bi,:]=[(patch.shape[0]-depth//2)/patch.shape[0],(j*half_stride+width//2)/patch.shape[1],(k*half_stride+height//2)/patch.shape[2]]
                bi=bi+1
                if bi==batch_size or j==pad_patch_shape[1]//half_stride-2 and k==pad_patch_shape[2]//half_stride-2:
                    test_out = cnn.predict([c_patch,patchpos])
                    #test_out = np.ones((batch_size,depth,width,height,1))
                    for bbi in range(bi):
                        cwi = patchloc[bbi,1]
                        chi = patchloc[bbi,2]
                        predict_img[(patch.shape[0]//depth)*depth : patch.shape[0],
                                   cwi * half_stride : cwi*half_stride+width,
                                   chi *half_stride: chi*half_stride+height]+=test_out[bbi,:].reshape(depth,width,height)[(patch.shape[0]//depth)*depth-patch.shape[0]:]
                    #reset
                    bi=0
        predict_img[pend:patch.shape[0],half_stride:-half_stride,half_stride:-half_stride]/=4
        predict_img[pend:patch.shape[0],:half_stride,half_stride:-half_stride]/=2
        predict_img[pend:patch.shape[0],-half_stride:,half_stride:-half_stride]/=2
        predict_img[pend:patch.shape[0],half_stride:-half_stride,:half_stride]/=2
        predict_img[pend:patch.shape[0],half_stride:-half_stride,-half_stride:]/=2

    predict_img=predict_img[0:patch.shape[0],0:patch.shape[1],0:patch.shape[2]]
    return predict_img

def createnormLUT(meanper,cper):
    NormLUT=[]
    for i in range(10):
        for j in range(int(cper[i]),int(cper[i+1])):
            skope = (meanper[i+1]-meanper[i])/(cper[i+1]-cper[i])
            NormLUT.append(int(round(meanper[i]+skope*(j-cper[i]))))
    for j in range(int(cper[10]),int(cper[11])+1):
        NormLUT.append(int(round(meanper[10]+skope*(j-cper[10]))))
    return NormLUT 

def Stopolar(car_img, rsamples = 0, thsamples = 180):
    if rsamples==0:
        rsamples = car_img.shape[0]//2
    if len(car_img.shape)==2:
        cimg = car_img[:,:,None]
    elif len(car_img.shape)==3:
        cimg = car_img
    else:
        print('channel not 2/3')
        return

    SUBTH = 360/thsamples
    height,width,channel=cimg.shape

    y = np.linspace(0, cimg.shape[0]-1, cimg.shape[0])
    x = np.linspace(0, cimg.shape[1]-1, cimg.shape[1])
    my_interpolating_function = RegularGridInterpolator((y, x), cimg)

    rth = np.zeros((thsamples,rsamples,channel))
    for th in range(thsamples):
        for r in range(rsamples):
            inty = cimg.shape[0]//2+r*math.sin(th*SUBTH/180*math.pi)
            intx = cimg.shape[1]//2+r*math.cos(th*SUBTH/180*math.pi)
            if intx>=cimg.shape[1]-1 or inty>=cimg.shape[0]-1:
                rth[th,r] = 0
            elif intx<0 or inty<0:
                rth[th,r] = 0
            else:    
                rth[th,r] = my_interpolating_function([inty,intx])

    if len(car_img.shape)==2:
        return rth[:,:,0]
    else:
        return rth

def topolar(car_img, rsamples = 0, thsamples = 180, intmethod = 'linear'):
    #BUG in cubic
    if rsamples==0:
        rsamples = car_img.shape[0]//2
    if len(car_img.shape)==2:
        cimg = car_img[:,:,None]
    elif len(car_img.shape)==3:
        cimg = car_img
    else:
        print('channel not 2/3')
        return

    SUBTH = 360/thsamples
    height,width,channel=cimg.shape

    grid = UCGrid((0, cimg.shape[1]-1, cimg.shape[1]), (0, cimg.shape[0]-1, cimg.shape[0]))

 
    # filter values
    if intmethod == 'cubic':
        coeffs = filter_cubic(grid, cimg) 
    
    rth = np.zeros((thsamples,rsamples,channel))
    for th in range(thsamples):
        for r in range(rsamples):
            inty = cimg.shape[0]//2+r*math.sin(th*SUBTH/180*math.pi)
            intx = cimg.shape[1]//2+r*math.cos(th*SUBTH/180*math.pi)
            if intx>=cimg.shape[1]-1 or inty>=cimg.shape[0]-1:
                rth[th,r] = 0
            elif intx<0 or inty<0:
                rth[th,r] = 0
            else:    
                if intmethod == 'cubic':
                    rth[th,r] = eval_cubic(grid, coeffs, np.array([inty,intx]))
                elif intmethod == 'linear':
                    rth[th,r] = eval_linear(grid, cimg, np.array([inty,intx]))

    if len(car_img.shape)==2:
        return rth[:,:,0]
    else:
        return rth

def Stocart(polar_img, rheight = 0, rwidth = 0):
    if rheight==0 or rwidth==0:
        rheight = polar_img.shape[1]*2
        rwidth = polar_img.shape[1]*2
    #transfer to cartesian based
    if len(polar_img.shape)==2:
        cimg = polar_img[:,:,None]
    elif len(polar_img.shape)==3:
        cimg = polar_img
    else:
        print('channel not 2/3')
        return
    rth,rr,rchannel = cimg.shape
    SUBTH = 360/rth
    x = np.linspace(0, cimg.shape[0]-1, cimg.shape[0])
    y = np.linspace(0, cimg.shape[1]-1, cimg.shape[1])
    my_interpolating_function = RegularGridInterpolator((x, y), cimg)

    test_out_c = np.zeros((rheight,rwidth,rchannel))
    for h in range(rheight):
        for w in range(rwidth):
            hy = h-rheight//2
            wx = w-rwidth//2
            intradius = int(np.sqrt(hy*hy+wx*wx))
            cth = math.atan2(hy,wx)/np.pi*180
            if cth<0:
                intth = (360+cth)
            else:
                intth = (cth)

            intth /= SUBTH
            if intth>cimg.shape[0]-1:
                intth = cimg.shape[0]-1
            #print(intradius,intth)
            if intradius>=cimg.shape[1]:
                test_out_c[h,w] = 0
            else:
                test_out_c[h,w] = my_interpolating_function([intth,intradius])    
    if len(polar_img.shape)==2:
        return test_out_c[:,:,0]
    else:
        return test_out_c

def tocart(polar_img, rheight = 0, rwidth = 0, intmethod = 'linear'):
    if rheight==0 or rwidth==0:
        rheight = polar_img.shape[1]*2
        rwidth = polar_img.shape[1]*2
    #transfer to cartesian based
    if len(polar_img.shape)==2:
        cimg = polar_img[:,:,None]
    elif len(polar_img.shape)==3:
        cimg = polar_img
    else:
        print('channel not 2/3')
        return
    rth,rr,rchannel = cimg.shape
    SUBTH = 360/rth
    
    grid = UCGrid((0, cimg.shape[0]-1, cimg.shape[0]), (0, cimg.shape[1]-1, cimg.shape[1]))

    test_out_c = np.zeros((rheight,rwidth,rchannel))
    for h in range(rheight):
        for w in range(rwidth):
            hy = h-rheight//2
            wx = w-rwidth//2
            intradius = int(np.sqrt(hy*hy+wx*wx))
            cth = math.atan2(hy,wx)/np.pi*180
            if cth<0:
                intth = (360+cth)
            else:
                intth = (cth)

            intth /= SUBTH
            if intth>cimg.shape[0]-1:
                intth = cimg.shape[0]-1
            #print(intradius,intth)
            if intradius>=cimg.shape[1]:
                test_out_c[h,w] = 0
            else:
                test_out_c[h,w] = eval_linear(grid, cimg, np.array([intth,intradius]))
    if len(polar_img.shape)==2:
        return test_out_c[:,:,0]
    else:
        return test_out_c

def Etocart(polar_img,rheight = 0, rwidth = 0):
    if rheight==0 or rwidth==0:
        rheight = polar_img.shape[1]*2
        rwidth = polar_img.shape[1]*2
    rmax = np.max(polar_img.shape[1]-np.argmax(polar_img[:,::-1]>0,axis=1))
    fsize = rheight
    carsegE = np.zeros((fsize,fsize))
    carsegE[fsize//2-rmax:fsize//2+rmax,fsize//2-rmax:fsize//2+rmax] = tocart(polar_img[:,:rmax],rmax*2,rmax*2)
    return carsegE

def fillpatch(srcimg,patchimg,cty=-1,ctx=-1):
    sheight = patchimg.shape[0]//2
    swidth = patchimg.shape[1]//2
    sheightRem = patchimg.shape[0] - patchimg.shape[0]//2
    swidthRem = patchimg.shape[1] - patchimg.shape[1]//2

    fillimg = srcimg.copy()
    inputheight = srcimg.shape[0]
    inputwidth = srcimg.shape[1]
    patchheight = patchimg.shape[0]
    patchwidth = patchimg.shape[1]

    if cty==-1:
        cty = inputheight//2
    if ctx==-1:
        ctx = inputwidth//2
    ctx = int(round(ctx))
    cty = int(round(cty))

    if ctx-swidth<0:
        p1 = 0
        r1 = -(ctx-swidth)
    else:
        p1 = ctx-swidth
        r1 = 0
    if ctx+swidthRem>inputwidth:
        p2 = inputwidth
        r2 = (ctx+swidthRem)-inputwidth
    else:
        p2 = ctx+swidthRem
        r2 = 0
    if cty-sheight<0:
        p3 = 0
        r3 = -(cty-sheight)
    else:
        p3 = cty-sheight
        r3 = 0
    if cty+sheightRem>inputheight:
        p4 = inputheight
        r4 = (cty+sheightRem)-inputheight
    else:
        p4 = cty+sheightRem
        r4 = 0
    #print(p1,p2,p3,p4,r1,r2,r3,r4)
    fillimg[p3:p4,p1:p2] = patchimg[r3:patchheight-r4,r1:patchwidth-r2]

    return fillimg

def croppatch3d(cartimgori,cty=-1,ctx=-1,ctz=-1,sheight=8,swidth=8,sdepth=8):
    if ctz==-1:
        ctz = cartimgori.shape[2]//2
    assert ctx<cartimgori.shape[0]
    assert cty<cartimgori.shape[1]
    assert ctz<cartimgori.shape[2]
    ctz = int(round(ctz))
    cartpatch = croppatch(cartimgori,cty=cty,ctx=ctx,sheight=sheight,swidth=swidth)
    if ctz<sdepth:
        padcartpatch = np.zeros((sheight*2,swidth*2,sdepth*2))
        padcartpatch[:,:,sdepth-ctz:] = cartpatch[:,:,:ctz+sdepth]
        return padcartpatch
    elif ctz>cartimgori.shape[2]-sdepth:
        padcartpatch = np.zeros((sheight*2,swidth*2,sdepth*2))
        padcartpatch[:,:,:sdepth+cartimgori.shape[2]-ctz] = cartpatch[:,:,ctz-sdepth:]
        return padcartpatch
    else:
        return cartpatch[:,:,ctz-sdepth:ctz+sdepth]

import copy
def croppatch(cartimgori,cty=-1,ctx=-1,sheight=40,swidth=40):
    cartimgori = copy.copy(cartimgori)
    def croppatch3(cartimgori,cty=-1,ctx=-1,sheight=40,swidth=40):
        #input height, width, (channel) large image, and a patch center position (cty,ctx)
        #output sheight, swidth, (channel) patch with padding zeros
        sheight = int(round(sheight))
        swidth = int(round(swidth))
        patchheight = sheight*2
        patchwidth = swidth*2
        if len(cartimgori.shape)<2:
            print('Not enough dim')
            return
        elif len(cartimgori.shape)==2:
            cartimg = cartimgori[:,:,None]
        elif len(cartimgori.shape)==3:
            cartimg = cartimgori
        elif len(cartimgori.shape)>3:
            print('Too many dim')
            return

        patchchannel = cartimg.shape[2]

        inputheight = cartimg.shape[0]
        inputwidth = cartimg.shape[1]
        #if no center point defined, use mid of cartimg
        if cty==-1:
            cty = inputheight//2
        if ctx==-1:
            ctx = inputwidth//2
        ctx = int(round(ctx))
        cty = int(round(cty))
        if ctx-swidth>cartimgori.shape[1] or cty-sheight>cartimgori.shape[0]:
            print('center outside patch')
            cartimgcrop = np.zeros((patchheight, patchwidth, patchchannel))
            return cartimgcrop
        #crop start end position
        if ctx-swidth<0:
            p1 = 0
            r1 = -(ctx-swidth)
        else:
            p1 = ctx-swidth
            r1 = 0
        if ctx+swidth>inputwidth:
            p2 = inputwidth
            r2 = (ctx+swidth)-inputwidth
        else:
            p2 = ctx+swidth
            r2 = 0
        if cty-sheight<0:
            p3 = 0
            r3 = -(cty-sheight)
        else:
            p3 = cty-sheight
            r3 = 0
        if cty+sheight>inputheight:
            p4 = inputheight
            r4 = (cty+sheight)-inputheight
        else:
            p4 = cty+sheight
            r4 = 0
        cartimgcrop = cartimg[p3:p4,p1:p2]
        #if not enough to extract, pad zeros at end
        if cartimgcrop.shape!=(patchheight,patchwidth,patchchannel):
            #print('Label Extract region out of border',p1,p2,p3,p4,r1,r2,r3,r4)
            cartimgcropc = cartimgcrop.copy()
            cartimgcrop = np.zeros((patchheight,patchwidth,patchchannel))
            cartimgcrop[r3:patchheight-r4,r1:patchwidth-r2] = cartimgcropc

        if len(cartimgori.shape)==2:
            return cartimgcrop[:,:,0]
        else:
            return cartimgcrop

        return cartimgcrop

    if len(cartimgori.shape)<4:
        return croppatch3(cartimgori,cty,ctx,sheight,swidth)
    elif len(cartimgori.shape)>4:
        print('Too many dim')
        return
    else:
        inputdepth = cartimgori.shape[2]
        cartimgcrop = np.zeros((sheight*2,swidth*2,inputdepth,cartimgori.shape[3]))
        for dpi in range(inputdepth):
            cartimgcrop[:,:,dpi,:] = croppatch3(cartimgori[:,:,dpi,:],cty,ctx,sheight,swidth)
        return cartimgcrop

def dist(pt1,pt2):
    assert len(pt1) == len(pt2)
    return np.sqrt(np.sum([pow((pt1[dim]-pt2[dim]),2) for dim in range(len(pt1))]))

def mergechannel(imgs):
    if len(imgs)>3:
        print('more than 3 channels')
        return
    mergeimg = np.zeros((imgs[0].shape[0],imgs[0].shape[1],3))
    mergeimg[:,:,0] = imgs[0]
    for chani in range(1,len(imgs)):
        mergeimg[:,:,chani] = imgs[chani]
    return mergeimg

