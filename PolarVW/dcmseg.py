from PolarVW.UTL import croppatch, topolar
from PolarVW.mindist import multi_min_dist_pred_withinbb
from PolarVW.db import crop_dcm_stack
from PolarVW.polarutil import polar_pred_cont_cst, toctbd, plotct, plotpolar
from PolarVW.eval import SegResult
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import datetime

def stack_polar_seg(vwcnn,min_dist_cnn,bbs,predname, einame, dicomstack, usegrad = False, iterpred = True, multiseg = False, cartreg = False, SCALE = 4):
    #cart position 256 points, first lumen second outer wall
    segresult = SegResult(predname, einame, dicomstack, load=False)
    DEBUG = 0

    vwcfg = {}
    vwcfg['patchheight'] = int(vwcnn.input.shape[1])
    vwcfg['height'] = int(vwcnn.input.shape[1])
    vwcfg['width'] = int(vwcnn.input.shape[2])
    vwcfg['depth'] = int(vwcnn.input.shape[3])
    vwcfg['channel'] = int(vwcnn.input.shape[4])

    for slicei in range(len(bbs)):
        #if slicei!=7:continue
        if len(bbs[slicei])==0:
            continue
        starttime = datetime.datetime.now()

        bb = bbs[slicei][0]
        cts, nms_dist_map_p = multi_min_dist_pred_withinbb(min_dist_cnn, dicomstack[slicei], bb, 64, 64)
        cts_dcm = to_dcm_cord(cts, bb)
        segresult.add_nms_cts(slicei, cts_dcm)

        if DEBUG:
            plt.imshow(nms_dist_map_p)
            plt.show()

        if len(cts_dcm)>1:
            print('multiple conts',len(cts_dcm))
        #cont position in cartesian 512 dcm cordinate
        if len(cts_dcm) == 0:
            if multiseg == True:
                continue
            print('no max in mindist, use center')
            cts_dcm.append([bb.x,bb.y])
        elif len(cts_dcm)!=1:
            if multiseg == False:
                print('smallest from ct',cts_dcm)
                dsts = [abs(ct[0]-256)+abs(ct[1]-256) for ct in cts_dcm]
                cts_dcm = [cts_dcm[np.argmin(dsts)]]
                #cts_dcm = [np.mean(cts_dcm,axis=0)]
            else:
                if len(cts_dcm)>2:
                    dsts = [abs(ct[0] - 256) + abs(ct[1] - 256) for ct in cts_dcm]
                    dst_arg_sort =np.argsort(dsts)
                    cts_dcm = [cts_dcm[dst_arg_sort[0]],cts_dcm[dst_arg_sort[1]]]
                    print('cts_dcm more than 2, reduce to',cts_dcm)
                else:
                    print('select ct', cts_dcm)
        elif len(cts_dcm)==1:
            if multiseg == True:
                continue

        contours, polarconsistency, cts_dcm_recenter = polar_seg_slice(vwcnn,vwcfg,dicomstack,slicei,cts_dcm,SCALE,usegrad,iterpred,cartreg)
        elaspsetime = datetime.datetime.now() - starttime
        cet = elaspsetime.total_seconds()
        segresult.additem(slicei, 'et', cet)
        segresult.addconts(slicei,contours,polarconsistency)
        segresult.addcts(slicei, cts_dcm_recenter)

    return segresult

from PolarVW.BB import BB
#from roi coordinate (vw centered in the img of 128*SCALE) to dicom coordinate (512*512)
def to_dcm_cord(cts,bb,SCALE=4):
    cts_dcm = []
    #cts_valid = ct_inside_bb(cts, bb)
    for ct in cts:
        if type(ct) in [list,np.ndarray]:
            cts_dcm.append([(ct[0]-256)/SCALE+bb.x,(ct[1]-256)/SCALE+bb.y])
        else:
            #BB format
            ctdcm = copy.copy(ct)
            ctdcm.x = (ct.x - 256) / SCALE + bb.x
            ctdcm.y = (ct.y - 256) / SCALE + bb.y
            ctdcm.w /= SCALE
            ctdcm.h /= SCALE
            cts_dcm.append(ctdcm)
    return cts_dcm

#from dcm coordinates to roi cordinates
def to_roi_cord(contours,bb,SCALE=4):
    #contours is a list of [x, y]
    contour_roi = []
    for cti in range(len(contours)):
        roix = (contours[cti][0] - bb.x) * SCALE + 256
        roiy = (contours[cti][1] - bb.y) * SCALE + 256
        contour_roi.append([roix, roiy])
    return contour_roi


from PolarVW.mergecont import mergecont
def polar_seg_slice(vwcnn,vwcfg,dicomstack,slicei,cts,SCALE = 4, usegrad = False, ITERPRED = True, cartreg=False):
    DEBUG = 0
    contourin = None
    contourout = None
    polarconsistency = None
    recenters = None
    if DEBUG:
        print(slicei,'cts',cts)
    if len(cts) == 1:
        ct = cts[0]
        ctx = ct[0]
        cty = ct[1]
        if ITERPRED == False:
            cartstack = crop_dcm_stack(dicomstack, slicei, cty, ctx, 64, 1)
            cartstack = cartstack / np.max(cartstack)
            polarimg = topolar(cartstack, 64, 64)
            polar_img_rz = cv2.resize(polarimg, (0, 0), fx=SCALE, fy=SCALE)
            if cartreg:
                cart_stack_rz = cv2.resize(cartstack, (0, 0), fx=2, fy=2)
                contbd = vwcnn.predict(cart_stack_rz[None, :, :, :, None])[0] * 256
                contour_in_roi = contbd[:,:2]+256
                contour_out_roi = contbd[:,2:]+256
                contourin = to_dcm_cord(contour_in_roi, BB(ctx,cty,0,0))
                contourout = to_dcm_cord(contour_out_roi, BB(ctx,cty,0,0))
                polarconsistency = None
            else:
                polarbd, polarsd = polar_pred_cont_cst(polar_img_rz[:, :, :, None], vwcfg, vwcnn, usegrad=usegrad)
                polarconsistency = 1 - np.mean(polarsd, axis=0)
                # contour positions [x, y] in original dicom space
                contourin, contourout = toctbd(polarbd / SCALE, ctx, cty)
            if DEBUG:
                contourin, contourout = toctbd(polarbd, 256, 256)
                cartvw = plotct(512, contourin, contourout)
                plt.subplot(1, 2, 1)
                cartstackdsp = copy.copy(cartstack)
                cartstackdsp[:, cartstackdsp.shape[1] // 2] = np.max(cartstackdsp)
                cartstackdsp[cartstackdsp.shape[0] // 2] = np.max(cartstackdsp)
                plt.imshow(cartstackdsp)
                plt.subplot(1, 2, 2)
                cartvwdsp = copy.copy(cartvw)
                cartvwdsp[:, cartvwdsp.shape[1] // 2] = np.max(cartvwdsp)
                cartvwdsp[cartvwdsp.shape[0] // 2] = np.max(cartvwdsp)
                plt.imshow(cartvwdsp)
                plt.show()
            # offset in 512 dcm cordinate

        else:
            # iterative pred and refine center
            REPS = 20
            MOVEFACTOR = 0.5
            MAX_OFF_STEP = 1
            mincty = cty
            minctx = ctx
            mindiff = np.inf
            lastoffset = [0,0]
            if DEBUG:
                print('init',ctx, cty)
            for repi in range(REPS):
                if repi == REPS-1 or MAX_OFF_STEP<0.1:
                    if DEBUG:
                        print('use min diff cts',minctx,mincty,'with mindiff',mindiff)
                    cty = mincty
                    ctx = minctx
                cartstack = crop_dcm_stack(dicomstack, slicei, cty, ctx, 64, 1)
                cartstack = cartstack/np.max(cartstack)
                polarimg = topolar(cartstack, 64, 64)
                polar_img_rz = cv2.resize(polarimg,(0,0),fx = SCALE, fy = SCALE)
                polarbd, polarsd = polar_pred_cont_cst(polar_img_rz[:,:,:,None], vwcfg, vwcnn, usegrad=usegrad)
                if DEBUG:
                    contourin, contourout = toctbd(polarbd,256,256)
                    cartvw = plotct(512, contourin, contourout)
                    plt.subplot(1,2,1)
                    cartstackdsp = copy.copy(cartstack)
                    cartstackdsp[:,cartstackdsp.shape[1]//2] = np.max(cartstackdsp)
                    cartstackdsp[cartstackdsp.shape[0] // 2] = np.max(cartstackdsp)
                    plt.imshow(cartstackdsp)
                    plt.subplot(1,2,2)
                    cartvwdsp = copy.copy(cartvw)
                    cartvwdsp[:,cartvwdsp.shape[1]//2] = np.max(cartvwdsp)
                    cartvwdsp[cartvwdsp.shape[0] // 2] = np.max(cartvwdsp)
                    plt.imshow(cartvwdsp)
                    plt.show()
                #offset in 512 dcm cordinate
                polar_cont_offset = cal_polar_offset(polarbd)
                if repi>0 and MAX_OFF_STEP>0.1 and lastoffset==polar_cont_offset:
                    #MAX_OFF_STEP += 0.5/SCALE
                    MAX_OFF_STEP = 0.09
                    if DEBUG:
                        print('move ct no change',MAX_OFF_STEP,lastoffset,polar_cont_offset)
                    continue
                cofftype = [polar_cont_offset[0]>0,polar_cont_offset[1]>0]
                if repi == 0:
                    offtype = cofftype

                if repi>2 and cofftype != offtype:
                    MAX_OFF_STEP /= 2
                    if DEBUG:
                        print('reduce max move to',MAX_OFF_STEP)
                offtype = cofftype
                polarconsistency = 1 - np.mean(polarsd, axis=0)
                #ccstl = polarconsistency[0]
                #ccstw = polarconsistency[1]
                # contour positions [x, y] in original dicom space
                contourin, contourout = toctbd(polarbd / SCALE, ctx, cty)
                cdif = np.max(abs(np.array(polar_cont_offset)))
                if cdif < 1 or MAX_OFF_STEP<0.1:
                    if cdif < 1:
                       print(polar_cont_offset)
                    print('==',slicei,'Break tracklet ref',polar_cont_offset)
                    break
                if cdif < mindiff:
                    mindiff = cdif
                    mincty = cty
                    minctx = ctx
                cofx = polar_cont_offset[0]
                cofy = polar_cont_offset[1]
                if abs(polar_cont_offset[0]) < 1:
                    cofx = 0
                if abs(polar_cont_offset[1]) < 1:
                    cofy = 0

                if repi<2:
                    ctx += cofx * MOVEFACTOR
                    cty += cofy * MOVEFACTOR
                else:
                    ctx = ctx + max(-MAX_OFF_STEP,min(MAX_OFF_STEP, cofx * MOVEFACTOR))
                    cty = cty + max(-MAX_OFF_STEP,min(MAX_OFF_STEP, cofy * MOVEFACTOR))
                print('repeat',repi,'offset',polar_cont_offset,ctx,cty)
                lastoffset = polar_cont_offset
        recenters = [[ctx,cty]]
    else:
        print('multiple cts',cts)
        all_contour_in = []
        all_contour_out = []
        all_polarconsistency = []
        for ct in cts:
            ctx = ct[0]
            cty = ct[1]
            cartstack = crop_dcm_stack(dicomstack, slicei, cty, ctx, 64, 1)
            cartstack = cartstack/np.max(cartstack)
            polarimg = topolar(cartstack, 64, 64)
            polar_img_rz = cv2.resize(polarimg, (0, 0), fx=SCALE, fy=SCALE)
            polarbd, polarsd = polar_pred_cont_cst(polar_img_rz[:, :, :, None], vwcfg, vwcnn)
            polarconsistency_c = 1 - np.mean(polarsd, axis=0)
            contour_in_c, contour_out_c = toctbd(polarbd / SCALE, ctx, cty)
            if DEBUG:
                plt.subplot(1,2,1)
                plt.imshow(polarimg)
                plt.subplot(1,2,2)
                polarseg = plotpolar(256, polarbd)
                plt.imshow(polarseg)
                plt.show()
            all_contour_in.append(contour_in_c)
            all_contour_out.append(contour_out_c)
            all_polarconsistency.append(polarconsistency_c)

        contourin = mergecont(all_contour_in, cts)
        contourout = mergecont(all_contour_out, cts)
        ctx, cty = np.mean(cts, axis=0)
        if DEBUG:
            seg_vw_merge = plotct(512 * SCALE, contourin * SCALE, contourout * SCALE)
            predseg = croppatch(seg_vw_merge, cty * SCALE, ctx * SCALE, 64 * SCALE, 64 * SCALE)
            plt.imshow(predseg)
            plt.show()
        polarconsistency = np.mean(np.array(all_polarconsistency),axis=0)
        #no center adjustment for multi ct prediction
        recenters = [[ctx, cty]]
    return [contourin, contourout], polarconsistency, recenters


def cal_polar_offset(polarct, SCALE = 4):
    n = len(polarct[:,0])
    axisdiff = [abs(polarct[i,0]-polarct[i+n//2,0]) for i in range(n//2)]
    maxrot = np.argmax(axisdiff)
    maxdeg = maxrot/(n/2)*180
    diffdist = polarct[maxrot,0]-polarct[maxrot+n//2,0]
    difx = np.cos(maxdeg / 180 * np.pi)
    dify = np.sin(maxdeg / 180 * np.pi)
    print('max rotation at deg',maxdeg,difx,dify)
    offsetx = diffdist * difx / SCALE
    offsety = diffdist * dify / SCALE
    return [offsetx,offsety]

