import numpy as np
import os
from src.db import load_dcm_stack
from src.Tracklet import Tracklet
from src.BB import BB

def gentracklets(bbs):
    Ltracklet = []
    Rtracklet = []
    for slicei in range(len(bbs)):
        if len(bbs[slicei])==0:
            Ltracklet.append([])
            Rtracklet.append([])
            continue
        elif len(bbs[slicei])==1:
            if bbs[slicei][0][0]<256:
                Ltracklet.append([])
                Rtracklet.append(bbs[slicei][0])
            else:
                Ltracklet.append(bbs[slicei][0])
                Rtracklet.append([])
        elif len(bbs[slicei])==2:
            if bbs[slicei][0][0]<bbs[slicei][1][0]:
                Rtracklet.append(bbs[slicei][0])
                Ltracklet.append(bbs[slicei][1])
            else:
                Ltracklet.append(bbs[slicei][0])
                Rtracklet.append(bbs[slicei][1])
        else:
            print('More than 2 bb',slicei)
            continue
    return [Ltracklet,Rtracklet]

def plotslicebb(predstack,bbs):
    reppredstack = predstack.copy()
    for slicei in range(predstack.shape[0]):
        for bbi in range(len(bbs[slicei])):
            xcenter = int(round(bbs[slicei][bbi][0]))
            ycenter = int(round(bbs[slicei][bbi][1]))
            xmin = int(round(bbs[slicei][bbi][0]-2))
            xmax = int(round(bbs[slicei][bbi][0]+2))
            ymin = int(round(bbs[slicei][bbi][1]-2))
            ymax = int(round(bbs[slicei][bbi][1]+2))
            reppredstack[slicei,ymin:ymax,xcenter-1:xcenter+2,0] = 255
            reppredstack[slicei,ycenter-1:ycenter+2,xmin:xmax,0] = 255
    return reppredstack

def locdcm(einame,dcmpath,yolom,predname):
    dicomstack = load_dcm_stack(dcmpath)
    pred_loc_filename = os.path.join(predname, einame + '_loc.npy')
    if os.path.exists(pred_loc_filename):
        tracklets = np.load(pred_loc_filename)
    else:
        bbr, predlocstack = yolom.predstack(dicomstack)
        # tracklet refinement
        Tbb = Tracklet(bbr, dicomstack)
        # Tbb.settrackletreffigpath(trackletimgpath)
        config = {}
        config['ba'] = 1
        config['pn'] = 2
        config['medks'] = 5
        Tbb.refbb(config)
        bbs = Tbb.getbbs()
        # predlocstack: depth, 512, 512, RGB
        reppredstack = plotslicebb(predlocstack, bbs)
        tracklets = gentracklets(bbs)
        np.save(pred_loc_filename, tracklets)
    return dicomstack,tracklets

from collections import Counter
def calseqproperty(bbs, seqbbsim, neckxcenter):
    ARTTYPE = ['ICAL', 'ECAL', 'ICAR', 'ECAR']
    bblabel = []  # record label of bb, same size and structure as bbs
    # give rough labels for each slice by position
    for slicei in range(len(bbs)):
        cbbs = bbs[slicei]
        cbblabel = np.ones((len(cbbs))) * (-1)
        xpos = []
        ypos = []
        confs = []
        Lobj = []
        Robj = []
        for bbir in cbbs:
            bbi = bbir.getminmaxclabel()
            xpos.append((bbi[0] + bbi[2]) / 2)
            ypos.append((bbi[1] + bbi[3]) / 2)
            confs.append(bbi[4])
        # sort from smallest to largest
        argl = np.argsort(xpos)
        # decide left or right side. Left side of image is right artery
        # if len(argl)<3 or len(argl)>4:
        if len(argl):
            # according to x size
            for ai in range(len(argl)):
                # print(xpos[argl[ai]])
                if xpos[argl[ai]] < neckxcenter:
                    Robj.append(argl[ai])
                else:
                    Lobj.append(argl[ai])
        '''elif len(argl)==3:
            Robj.append(argl[0])
            if xpos[argl[1]]-xpos[argl[0]]<xpos[argl[2]]-xpos[argl[1]]:
                Robj.append(argl[1])
                Lobj.append(argl[2])
            else:
                Lobj.append(argl[1])
                Lobj.append(argl[2])
        elif len(argl)==4:
            Robj.append(argl[0])
            Robj.append(argl[1])
            Lobj.append(argl[2])
            Lobj.append(argl[3])'''

        # if more than 2 on one side
        if len(Lobj) > 2:
            Lconfs = []
            for ai in range(len(Lobj)):
                Lconfs.append(confs[Lobj[ai]])
            arglconf = np.argsort(Lconfs)
            Lobj = [Lobj[arglconf[0]]] + [Lobj[arglconf[1]]]
            print('Multiple bb on left side, top2', Lobj)
        if len(Robj) > 2:
            Rconfs = []
            for ai in range(len(Robj)):
                Rconfs.append(confs[Robj[ai]])
            argrconf = np.argsort(Rconfs)
            Robj = [Robj[argrconf[0]]] + [Robj[argrconf[1]]]
            print('Multiple bb on right side, top2', Robj)
        # assign label to bb
        if len(Lobj) == 2:
            if ypos[Lobj[1]] > ypos[Lobj[0]]:
                cbblabel[Lobj[0]] = ARTTYPE.index('ECAL')
                cbblabel[Lobj[1]] = ARTTYPE.index('ICAL')
            else:
                cbblabel[Lobj[0]] = ARTTYPE.index('ICAL')
                cbblabel[Lobj[1]] = ARTTYPE.index('ECAL')
        elif len(Lobj) == 1:
            cbblabel[Lobj[0]] = ARTTYPE.index('ICAL')
        elif len(Lobj) == 0:
            None
        else:
            print(slicei, 'Lobj', len(Lobj))

        if len(Robj) == 2:
            if ypos[Robj[1]] > ypos[Robj[0]]:
                cbblabel[Robj[0]] = ARTTYPE.index('ECAR')
                cbblabel[Robj[1]] = ARTTYPE.index('ICAR')
            else:
                cbblabel[Robj[0]] = ARTTYPE.index('ICAR')
                cbblabel[Robj[1]] = ARTTYPE.index('ECAR')
        elif len(Robj) == 1:
            cbblabel[Robj[0]] = ARTTYPE.index('ICAR')
        elif len(Robj) == 0:
            None
            # print('no')
        else:
            print(slicei, 'Robj', len(Robj))

        bblabel.append(cbblabel)

    # majority vote for artery label in each seq
    seqlabel = []
    seqx = []
    seqy = []
    seqc = []
    seqsz = []
    for seqi in range(len(seqbbsim)):
        if len(seqbbsim[seqi]) == 0:
            seqlabel.append(-1)
            print('seqbbsim', seqi, ' len 0')
            continue
        cx = []
        cy = []
        cc = []
        clabel = []
        seqsz.append(len(seqbbsim[seqi]))
        for bbi in range(len(seqbbsim[seqi])):
            slicei, bbid = seqbbsim[seqi][bbi]
            bb = bbs[slicei][bbid].getminmaxclabel()
            cx.append((bb[0] + bb[2]) / 2)
            cy.append((bb[1] + bb[3]) / 2)
            cc.append(bb[4])
            clabel.append(bblabel[slicei][bbid])
        seqx.append(np.mean(cx))
        seqy.append(np.mean(cy))
        seqc.append(np.mean(cc))
        if len(clabel) == 0:
            seqlabel.append(-1)
            print('seqbbsim', seqi, ' len 0')
        else:
            clabelct = Counter(clabel)
            seqlabel.append(int(clabelct.most_common(1)[0][0]))

    # check I/ECAL/R has only one at certain slicei
    seqlen = [len(seqi) for seqi in seqbbsim]
    # start from longest seq
    seqlenarg = np.array(seqlen).argsort()[::-1]
    # allow only one per type on each slice
    availart = np.ones((4, len(bbs)))

    # use bbs and seqlabel to update bb info all_artery_objs
    # start from longest
    for seqid in seqlenarg:
        for seqbbid in range(len(seqbbsim[seqid])):
            slicei, bbid = seqbbsim[seqid][seqbbid]
            # left side, take ICA then ECA, then skip
            if seqlabel[seqid] in [0, 1]:
                if availart[0][slicei] == 1:
                    artlabel = ARTTYPE[0]
                    availart[0][slicei] = 0
                elif availart[1][slicei] == 1:
                    artlabel = ARTTYPE[1]
                    availart[1][slicei] = 0
                else:
                    print(slicei, 'Full art bb on left side, skip seqid', seqid)
                    seqlabel[seqid] = -1
                    break
            if seqlabel[seqid] in [2, 3]:
                if availart[2][slicei] == 1:
                    artlabel = ARTTYPE[2]
                    availart[2][slicei] = 0
                elif availart[3][slicei] == 1:
                    artlabel = ARTTYPE[3]
                    availart[3][slicei] = 0
                else:
                    print(slicei, 'Full art bb on left side, skip seqid', seqid)
                    seqlabel[seqid] = -1
                    break
    return seqlabel


def identify_tracklets(bbs, seqbbsim, neckxcenter, dicomstack):
    maxslice = dicomstack.shape[0]
    bbtrack = [[] for i in range(maxslice)]
    seqbbtrack = []
    seqlabel = calseqproperty(bbs, seqbbsim, neckxcenter)
    seq_bb_num_l = [len(seqbbsim[i]) if seqlabel[i] == 0 else 0 for i in range(len(seqbbsim))]
    arg_seq_id_l = np.argsort(seq_bb_num_l)[::-1]
    seq_tracklet_l = [(slicei, 0) for slicei in range(maxslice)]
    seqbbtrack.append(seq_tracklet_l)
    for seqid in arg_seq_id_l:
        if seq_bb_num_l[seqid] == 0:
            continue
        for seqbbi in range(len(seqbbsim[seqid])):
            slicei = seqbbsim[seqid][seqbbi][0]
            bbid = seqbbsim[seqid][seqbbi][1]
            if len(bbtrack[slicei]) == 0:
                bbtrack[slicei].append(bbs[slicei][bbid])
            else:
                print('exist bb l ', slicei, len(bbtrack[slicei]))

    for slicei in range(len(bbtrack)):
        if len(bbtrack[slicei]) == 0:
            # search next non zero
            for exti in range(1, max(slicei, maxslice - slicei)):
                if slicei - exti >= 0 and len(bbtrack[slicei - exti]) == 1:
                    bbtrack[slicei].append(bbtrack[slicei - exti][0])
                    break
                if slicei + exti < maxslice and len(bbtrack[slicei + exti]) == 1:
                    bbtrack[slicei].append(bbtrack[slicei + exti][0])
                    break

    seq_bb_num_r = [len(seqbbsim[i]) if seqlabel[i] == 2 else 0 for i in range(len(seqbbsim))]
    arg_seq_id_r = np.argsort(seq_bb_num_r)[::-1]
    seq_tracklet_r = [(slicei, 1) for slicei in range(maxslice)]
    seqbbtrack.append(seq_tracklet_r)
    for seqid in arg_seq_id_r:
        if seq_bb_num_r[seqid] == 0:
            continue
        for seqbbi in range(len(seqbbsim[seqid])):
            slicei = seqbbsim[seqid][seqbbi][0]
            bbid = seqbbsim[seqid][seqbbi][1]
            if len(bbtrack[slicei]) == 1:
                bbtrack[slicei].append(bbs[slicei][bbid])
            else:
                print('exist bb r ', slicei, len(bbtrack[slicei]))

    for slicei in range(len(bbtrack)):
        if len(bbtrack[slicei]) == 1:
            # search next non zero
            for exti in range(1, max(slicei, maxslice - slicei)):
                if slicei - exti >= 0 and len(bbtrack[slicei - exti]) == 2:
                    bbtrack[slicei].append(bbtrack[slicei - exti][1])
                    break
                if slicei + exti < maxslice and len(bbtrack[slicei + exti]) == 2:
                    bbtrack[slicei].append(bbtrack[slicei + exti][1])
                    break
    return bbtrack, seqbbtrack

def collectbb(bbfolder, classnumber=1):
    if len(os.listdir(bbfolder)) == 0:
        print('no bb in folder', bbfolder)
        return
    bbr = [[] for i in range(len(os.listdir(bbfolder)))]
    bbfilelist = os.listdir(bbfolder)
    for slicepathi in range(len(os.listdir(bbfolder))):
        slicei = int(bbfilelist[slicepathi][:-4].split('_')[-1])
        slicename = os.path.join(bbfolder, bbfilelist[slicepathi])
        with open(slicename, 'r') as file:
            bbnum = int(file.readline()[:-1])
            if bbnum == 0:
                continue
            cbbs = []
            for bbi in range(bbnum):
                cbblist = [float(i) for i in file.readline()[:-1].split(' ')]
                cbblist[-1] = int(cbblist[-1])
                cbbs.append(BB.fromminmaxlistclabel(cbblist, classnumber))
            bbr[slicei].extend(cbbs)
    return bbr

from src.UTL import croppatch

def ct_inside_bb(cts, bb):
    cts_valid = []
    for ct in cts:
        #print(ct, bb)
        if abs(ct[0] - 256) < bb.w / 2 * 4 and abs(ct[1] - 256) < bb.h / 2 * 4:
            cts_valid.append(ct)
        else:
            print(ct, 'outside bb')
    return cts_valid


from src.mindist import multi_min_dist_pred,mergects
def find_ct_slice(min_dist_cnn, dicomstack, slicei, bball):
    meancts = []
    for bb in bball:
        cts, nms_dist_map_p = multi_min_dist_pred(min_dist_cnn, dicomstack[slicei], bb.y, bb.x, 64, 64)
        cts_valid = ct_inside_bb(cts, bb)
        meancts = mergects(nms_dist_map_p[:,:,0], cts_valid)
        # transform to original dcm 512 cordinate
        for meanct in meancts:
            meanct.x = (meanct.x - 256) / 4 + bb.x
            meanct.y = (meanct.y - 256) / 4 + bb.y
            meanct.w = meanct.w / 4
            meanct.h = meanct.h / 4
            meanct.c = bb.c
            meanct.classes = bb.classes
    #print(slicei, meancts)
    return meancts


def find_cts(min_dist_cnn, dicomstack, bbr):
    cts = [None for i in range(len(dicomstack))]
    for slicei in range(len(dicomstack)):
        cts[slicei] = find_ct_slice(min_dist_cnn, dicomstack, slicei, bbr[slicei])
    return cts

from src.UTL import dist
def findtrack(min_dist_cnn, caseloader, bbr, dicomstack):
    bbrside = [[] for i in range(len(bbr))]
    for slicei in range(len(bbr)):
        for bbi in bbr[slicei]:
            if bbi.x < 256 and caseloader.side == 'R' or \
                    bbi.x > 256 and caseloader.side == 'L':
                bbrside[slicei].append(bbi)
    tracklet = find_cts(min_dist_cnn, dicomstack, bbrside)
    for slicei in range(len(tracklet)):
        #print(tracklet[slicei])
        if slicei > 0 and len(tracklet[slicei]) > 1 and len(tracklet[slicei-1]) == 1:
            pct = tracklet[slicei-1][0]
            dsts = []
            for bbi in tracklet[slicei]:
                dsts.append(dist([pct.x,pct.y], [bbi.x,bbi.y]))
            print(slicei,dsts)
            tracklet[slicei] = [tracklet[slicei][np.argmin(dsts)]]
    return tracklet


import copy
#fill missing gaps by mean of neighbors
def fill_tracklet_gap(trackletr):
    tracklet = copy.deepcopy(trackletr)
    for slicei in range(len(trackletr)):
        slicebbs = trackletr[slicei]
        if len(slicebbs)==0:
            neighbbs = []
            for neid in range(1,3):
                if slicei-neid>=0 and len(trackletr[slicei-neid])==1:
                    neighbbs.append(trackletr[slicei-neid])
                if slicei+neid<len(trackletr) and len(trackletr[slicei+neid])==1:
                    neighbbs.append(trackletr[slicei+neid])
            if len(neighbbs):
                print('fill missing bb at',slicei,len(neighbbs))
                tracklet[slicei] = np.mean(neighbbs,axis=0).tolist()
            else:
                print('no neighbors at',slicei)
                for neid in range(1,len(trackletr)):
                    if slicei - neid >= 0 and slicei - neid < len(trackletr):
                        if len(tracklet[slicei-neid])>=1:
                            tracklet[slicei] = tracklet[slicei-neid]
                            print('add bb at',slicei,'from',slicei-neid)
                            break
                    if slicei + neid >= 0 and slicei + neid < len(trackletr):
                        if len(tracklet[slicei+neid])>=1:
                            tracklet[slicei] = tracklet[slicei+neid]
                            print('add bb at', slicei, 'from', slicei + neid)
                            break

    return tracklet

import matplotlib.pyplot as plt
def display_tracklet(tracklet,figfilename=None):
    featx = []
    featy = []
    slices = []
    for slicei in range(len(tracklet)):
        for bbi in tracklet[slicei]:
            featx.append(bbi.x)
            featy.append(bbi.y)
            slices.append(slicei)
    plt.title('tracklet pos')
    plt.plot(featx,slices,'o')
    plt.plot(featy, slices, 'o')
    plt.legend(['x','y'])
    if figfilename is not None:
        plt.savefig(figfilename)
    else:
        plt.show()
    plt.close()


def display_tracklet_dcm_patch(tracklet,dicomstack,figfilename=None):
    plt.figure(figsize=(18,5))
    for slicei in range(len(tracklet)):
        for bb in tracklet[slicei]:
            dcmslice = croppatch(dicomstack[slicei],bb.y,bb.x,bb.h,bb.w)
            w = int(round(bb.w/2))
            h = int(round(bb.h/2))
            dcmslice[h, w:w * 3] = np.max(dcmslice)
            dcmslice[h:h * 3, w] = np.max(dcmslice)
            dcmslice[h*3, w:w * 3] = np.max(dcmslice)
            dcmslice[h:h * 3, w*3] = np.max(dcmslice)
            plt.subplot(2,int(np.ceil(len(dicomstack)/2)),slicei+1)
            plt.imshow(dcmslice)
            plt.title(slicei)
    if figfilename is not None:
        plt.savefig(figfilename)
    else:
        plt.show()
    plt.close()


def intpath(pos1, pos2, dicomstack):
    DEBUG = 0
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos1int = [int(round(posi)) for posi in pos1]
    pos2int = [int(round(posi)) for posi in pos2]
    direction = pos2 - pos1
    dist = np.linalg.norm(direction)
    print(dist)
    dirnorm = direction / dist
    intp = []
    intp.append(dicomstack[pos1int[2]][pos1int[1]][pos1int[0]])
    if DEBUG:
        pdsp = croppatch(dicomstack[pos1int[2]], pos1int[1], pos1int[0])
        pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
        pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
        plt.title('start')
        plt.imshow(pdsp)
        plt.show()

    for stepi in range(int(np.floor(dist))):
        cpos = pos1 + dirnorm * stepi
        cposint = np.array([int(np.round(cposi)) for cposi in cpos])
        if DEBUG:
            pdsp = croppatch(dicomstack[cposint[2]], cpos[1], cpos[0])
            pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
            pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
            plt.imshow(pdsp)
            plt.title(str(cposint[2]))
            plt.show()

        intp.append(dicomstack[cposint[2]][cposint[1]][cposint[0]])
    intp.append(dicomstack[pos2int[2]][pos2int[1]][pos2int[0]])
    if DEBUG:
        pdsp = croppatch(dicomstack[pos2int[2]], pos2int[1], pos2int[0])
        pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
        pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
        plt.title('end')
        plt.imshow(pdsp)
        plt.show()
    return intp


def track_con_path(bbr, trackend, trackstart):
    bbend = bbr[trackend[0]][trackend[1]]
    bbstart = bbr[trackstart[0]][trackstart[1]]
    intensity_along_path = intpath([bbend.x, bbend.y, trackend[0]], [bbstart.x, bbstart.y, trackstart[0]], dicomstack)
    return intensity_along_path


#find match ct (with previous slice) from min dist map
def ct_with_min_change(bbprev,slicei,cts_dcm,dicomstack):
    minstd = np.inf
    ctm = cts_dcm[0]
    for ci in range(len(cts_dcm)):
        posstart = [bbprev.x,bbprev.y,slicei-1]
        posend = [cts_dcm[ci][0],cts_dcm[ci][1],slicei]
        path_int = intpath(posstart,posend,dicomstack)
        cstd = np.std(path_int)
        if cstd<minstd:
            minstd = cstd
            ctm = cts_dcm[ci]
    return ctm


from src.mindist import multi_min_dist_pred_withinbb
from src.dcmseg import to_dcm_cord
#continuous center with predicted bounding box
def cont_ct_within_bb(min_dist_cnn, dicomstack, slicei, bb, seq, bbprev):
    DEBUG = 1
    cts, nms_dist_map_p = multi_min_dist_pred_withinbb(min_dist_cnn, dicomstack[slicei] / np.max(dicomstack[slicei]),
                                                       bb, 64, 64)
    # merge multiple ct in the same connected region
    merge_cts = mergects(nms_dist_map_p[:, :, 0], cts)
    # convert ct to dicom coordinate
    cts_dcm = to_dcm_cord(merge_cts, bb)

    if DEBUG:
        plt.imshow(nms_dist_map_p)
        plt.show()
    if len(cts_dcm) == 1:
        bbx = cts_dcm[0][0]
        bby = cts_dcm[0][1]
    elif len(cts_dcm) > 1:
        # first slice use min to the center
        if slicei == seq[0][0]:
            bbx = cts_dcm[0][0]
            bby = cts_dcm[0][1]
        # other slices find minimum int change
        else:
            print('multiple', cts_dcm)
            ctm = ct_with_min_change(bbprev, slicei, cts_dcm)
            print('ctm', ctm)
            plt.imshow(croppatch(dicomstack[slicei], ctm[1], ctm[0]))
            plt.show()
            bbx = ctm[0]
            bby = ctm[1]
    else:
        print('no cts')

    return bbx, bby
