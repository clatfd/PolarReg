import copy
import matplotlib.pyplot as pyplot
import numpy as np
import scipy
from scipy import signal
import os
from collections import Counter

from PolarVW.BB import BB
from PolarVW.UTL import croppatch, fillpatch
from PolarVW.triplet import bb_match_score


def intpath(pos1, pos2, dicomstack):
    DEBUG = 0
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos1int = [int(round(posi)) for posi in pos1]
    if pos1int[0] < 0:
        pos1int[0] = 0
    if pos1int[0] > dicomstack.shape[1] - 1:
        pos1int[0] = dicomstack.shape[1] - 1
    if pos1int[1] < 0:
        pos1int[1] = 0
    if pos1int[1] > dicomstack.shape[0] - 1:
        pos1int[1] = dicomstack.shape[0] - 1

    pos2int = [int(round(posi)) for posi in pos2]
    direction = pos2 - pos1
    dist = np.linalg.norm(direction)
    dirnorm = direction / dist
    intp = []
    intp.append(dicomstack[pos1int[2]][pos1int[1]][pos1int[0]])
    if DEBUG:
        pdsp = croppatch(dicomstack[pos1int[2]], pos1int[1], pos1int[0])
        pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
        pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
        pyplot.title('start')
        pyplot.imshow(pdsp)
        pyplot.show()

    for stepi in range(1, int(np.floor(dist))):
        cpos = pos1 + dirnorm * stepi
        cposint = np.array([int(np.round(cposi)) for cposi in cpos])
        if DEBUG:
            pdsp = croppatch(dicomstack[cposint[2]], cpos[1], cpos[0])
            pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
            pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
            pyplot.imshow(pdsp)
            pyplot.title(str(cposint[2]))
            pyplot.show()

        valceil = dicomstack[int(np.ceil(cpos[2]))][cposint[1]][cposint[0]]
        valfloor = dicomstack[int(np.floor(cpos[2]))][cposint[1]][cposint[0]]
        w1 = int(np.ceil(cpos[2])) - cpos[2]
        w2 = cpos[2] - int(np.floor(cpos[2]))
        intp.append(valceil * w2 / (w1 + w2) + valfloor * w1 / (w1 + w2))

    intp.append(dicomstack[pos2int[2]][pos2int[1]][pos2int[0]])
    if DEBUG:
        pdsp = croppatch(dicomstack[pos2int[2]], pos2int[1], pos2int[0])
        pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
        pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
        pyplot.title('end')
        pyplot.imshow(pdsp)
        pyplot.show()
    return intp


class Tracklet:
    # self.seqbb and self.bbr store tracklet info
    # self.seqbbsim and self.bbs after refinement
    def __init__(self, bbr, classsize, imgstack=None, iouthres=0.65):
        self.RESOLUTION = -1
        self.iouexpandratio = 1.0
        self.imgstack = imgstack
        assert imgstack.shape[0] == len(bbr)
        self.bbr = bbr
        self.classsize = classsize
        # self.bbr = [[] for i in range(len(bbr))]
        # for slicei in range(len(bbr)):
        # 	for bi in range(len(bbr[slicei])):
        # 		x,y,w,h = [float(bbr[slicei][bi][i]) for i in range(4)]
        # 		classes = bbr[slicei][bi][4]

        # 		cbblist = BB(x,y,w,h,classes=classes)
        # 		self.bbr[slicei].append(cbblist)
        self.genseq(iouthres)
        self.bbs = None
        self.seqbbsim = None
        self.trackletreffigpath = None
        self.gwsegstack = None
        self.RESTRICT_BRANCH_NUM = 1

    def __repr__(self):
        rtstr = 'Tracklet seqs:%d' % (len(self.seqbb))
        if self.bbs is None:
            rtstr += ', unrefined'
        else:
            rtstr += ', refined to %d' % (len(self.seqbbsim))
        if self.trackletreffigpath is not None:
            rtstr += ', reffigpath: ' + self.trackletreffigpath
        return rtstr

    def genseq(self, iouthres):
        # continuous sequence, each seq a list with each pt recorded as sliceid, bb id on that slice
        self.seqbb = []
        lbb = []  # last slice bounding box
        lbbid = []  # last slice bb id
        bifseqid = []
        for slicei in range(len(self.bbr)):
            # set available bb in this slice
            availbbslicei = [1 for i in range(len(self.bbr[slicei]))]
            # check last slice whether has continous seq
            lbbi = 0
            while lbbi < len(lbb):
                # find max iou for last slice bb
                cmaxiou = 0
                cmaxbbi = -1
                for bbi in range(len(self.bbr[slicei])):
                    ioum = self.bbr[slicei][bbi].get_expand_iou(lbb[lbbi], self.iouexpandratio)
                    if ioum > cmaxiou:
                        cmaxiou = ioum
                        cmaxbbi = bbi
                if cmaxiou < iouthres:
                    # not found, stop seq
                    del lbb[lbbi]
                    del lbbid[lbbi]
                else:
                    # print(cmaxiou)
                    # update seqbb
                    self.seqbb[lbbid[lbbi]].append((slicei, cmaxbbi))
                    if availbbslicei[cmaxbbi] == 1:
                        availbbslicei[cmaxbbi] = 0
                        lbb[lbbi] = copy.deepcopy(self.bbr[slicei][cmaxbbi])
                        lbbi += 1
                    else:
                        # print('bif')
                        bifseqid.append(copy.deepcopy(lbbid[lbbi]))
                        del lbb[lbbi]
                        del lbbid[lbbi]

            # remaining bb set as new lbb
            for avi in range(len(availbbslicei)):
                if availbbslicei[avi] == 1:
                    # add as new lbb
                    lbb.append(copy.deepcopy(self.bbr[slicei][avi]))
                    lbbid.append(len(self.seqbb))
                    self.seqbb.append([(slicei, avi)])
        print('seq length', len(self.seqbb))

    def drawseqbb(self, dim='x', savename=None):
        self.connectbb_seqdraw(self.seqbb, self.bbr, dim, savename)

    def drawrfseqbb(self, dim='x', savename=None):
        if self.seqbbsim is None:
            print('Refine first')
            return
        self.connectbb_seqdraw(self.seqbbsim, self.bbs, dim, savename)

    def connectbb_seqdraw(self, seqbb, bb, dim, savename=None):
        for seqi in range(len(seqbb)):
            # print(seqi,len(seqbb[seqi]))
            sliceq = [seqbb[seqi][slicei][0] for slicei in range(len(seqbb[seqi]))]
            xseq = [getattr(bb[seqbb[seqi][slicei][0]][seqbb[seqi][slicei][1]], dim) for slicei in range(len(sliceq))]
            pyplot.plot(xseq, sliceq, 'o', label='Tracklet%d' % seqi)
        # pyplot.legend(loc="lower right")
        pyplot.legend()
        # pyplot.xlim([200,550])
        pyplot.ylabel('slice numbers', fontsize=18)
        pyplot.xlabel(dim + ' position', fontsize=18)
        pyplot.gca().invert_yaxis()
        if savename is not None:
            pyplot.savefig(savename)
        pyplot.show()

    def refbb(self, config=None):
        # pn: prior knowledge for fixed number of tracklets
        DEBUG = 0
        if config is None:
            pn = 1
            boundaryadjustment = 0
        else:
            if 'pn' in config:
                pn = config['pn']
            else:
                pn = 1
            if 'ba' in config:
                boundaryadjustment = config['ba']
            else:
                boundaryadjustment = 0
            if 'medks' in config:
                self.medks = config['medks']
            else:
                self.medks = 11
            if 'smoothfilterhsz' in config:
                self.smoothfilterhsz = config['smoothfilterhsz']
            else:
                self.smoothfilterhsz = 2
            if 'usebranch' in config:
                self.usebranch = config['usebranch']
            else:
                self.usebranch = 1
            if 'debug' in config:
                DEBUG = config['debug']

        self.seqbbsim = []
        seqbb = copy.deepcopy(self.seqbb)
        # starting/ending slice id of each seq
        startseqbb = [seqbb[i][0][0] for i in range(len(seqbb))]
        endseqbb = [seqbb[i][-1][0] for i in range(len(seqbb))]
        # [(i,startseqbb[i],endseqbb[i]) for i in range(len(seqbb))]

        weightiou = 0.5
        weightgap = 0.01
        weightbbfit = 0.001
        lossthres = 0.5
        connectend = []
        connectstart = []
        # find seqbb after end of each seqbb
        for seqi in range(len(seqbb)):
            # skip if already end of slice
            if endseqbb[seqi] == len(self.bbr):
                continue

            # already has connection
            if seqi in connectend:
                continue
            if DEBUG:
                print(seqi, self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].getbb())
                print(seqi, self.bbr[seqbb[seqi][-1][0]][seqbb[seqi][-1][1]].getbb())

            cminloss = np.inf
            seqminind = -1

            availseqj = [i for i in range(len(startseqbb)) if
                         startseqbb[i] >= endseqbb[seqi] and i not in connectstart + [seqi]]
            for seqj in availseqj:
                # find seqi end bb
                slicetuplei = seqbb[seqi][-1]
                bbi = copy.deepcopy(self.bbr[slicetuplei[0]][slicetuplei[1]])
                # find seqj start bb
                slicetuplej = seqbb[seqj][0]
                bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
                # calculate loss
                ciou = bbi.get_expand_iou(bbj, self.iouexpandratio)
                cgap = startseqbb[seqj] - endseqbb[seqi]
                diffx = abs(bbi.x - bbj.x)
                diffy = abs(bbi.y - bbj.y)
                if ciou == 0 and (diffx + diffy > 10) or cgap > 25:
                    # print(seqi,seqj,'ciou 0, dif',diffx,diffy)
                    continue
                cbbfit = bbi.get_bb_fit(bbj)

                # closs = (1-ciou)*weightiou + cbbfit*weightbbfit + cgap*weightgap
                closs = (1 - ciou) * weightiou + cbbfit * weightbbfit + cgap * weightgap
                if DEBUG:
                    print(seqi, '(%d,%d-%d,%d)' % (
                    seqbb[seqi][-1][0] - seqbb[seqi][0][0], seqbb[seqi][0][0], seqbb[seqi][-1][0],
                    self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].x),
                          seqj, '(%d,%d-%d,%d)' % (
                          seqbb[seqj][-1][0] - seqbb[seqj][0][0], seqbb[seqj][0][0], seqbb[seqj][-1][0],
                          self.bbr[seqbb[seqj][0][0]][seqbb[seqj][0][1]].x),
                          ciou, cbbfit, cgap * weightgap, closs)
                if closs < lossthres:
                    if closs < cminloss:
                        cminloss = closs
                        seqminind = seqj

            if seqminind == -1:
                # print('all start of seq not fit')
                continue
            else:
                # print('End of ',seqi,' to start of ',seqminind,' has min loss, test other ends')
                slicetuplej = seqbb[seqminind][0]
                bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
                ciminloss = np.inf
                seqiminind = -1
                availseqii = [i for i in range(len(endseqbb)) if
                              endseqbb[i] <= startseqbb[seqminind] and i not in connectend + [seqminind]]
                for seqii in availseqii:
                    # find seqii end bb
                    slicetupleii = seqbb[seqii][-1]
                    bbii = copy.deepcopy(self.bbr[slicetupleii[0]][slicetupleii[1]])
                    # calculate loss
                    ciou = bbii.get_expand_iou(bbj, self.iouexpandratio)
                    cgap = startseqbb[seqminind] - endseqbb[seqii]
                    diffx = abs(bbii.x - bbj.x)
                    diffy = abs(bbii.y - bbj.y)
                    if ciou == 0 and (diffx > 10 or diffy > 10) or cgap > 25:
                        continue
                    cbbfit = bbii.get_bb_fit(bbj)

                    closs = (1 - ciou) * weightiou + cbbfit * weightbbfit + cgap * weightgap
                    if DEBUG:
                        print('R', seqii, seqminind, ciou, cbbfit, cgap * weightgap, closs)
                    if closs < lossthres:
                        if closs < ciminloss:
                            ciminloss = closs
                            seqiminind = seqii
                if seqiminind == seqi:
                    if DEBUG:
                        print('back match, connect', seqi, seqminind)
                    # connect from i to seqminind
                    connectend.append(seqi)
                    connectstart.append(seqminind)
                else:
                    if DEBUG:
                        print('back mismatch', seqi, seqminind, seqminind)

        # deep copy
        bbint = copy.deepcopy(self.bbr)
        seqbbint = []
        # make connection with start not in connect start
        for seqi in range(len(seqbb)):
            if seqi in connectstart:
                continue
            seqbbint.append(copy.deepcopy(seqbb[seqi]))
            cseq = seqi
            while cseq in connectend:
                nextseq = connectstart[connectend.index(cseq)]
                if DEBUG:
                    print('connect', cseq, nextseq)
                bbend = bbint[seqbb[cseq][-1][0]][seqbb[cseq][-1][1]].getbbclassflat()
                bbstart = self.bbr[seqbb[nextseq][0][0]][seqbb[nextseq][0][1]].getbbclassflat()
                if DEBUG:
                    print('bbstart', bbstart, 'end', bbend)
                bbdiff = [bbstart[i] - bbend[i] for i in range(len(bbend))]
                gap = startseqbb[nextseq] - endseqbb[seqi]
                if DEBUG:
                    print('gap', gap, 'diff', bbdiff)
                for coni in range(endseqbb[seqi] + 1, startseqbb[nextseq]):
                    if DEBUG:
                        print('appending slice', coni)
                        print([bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))])
                    seqbbint[-1].append((coni, len(bbint[coni])))
                    intpbblist = [bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))]
                    bbint[coni].append(BB.fromlistclass(intpbblist))
                    if DEBUG:
                        print(len(seqbb[0]), 'cseq')
                seqbbint[-1].extend(seqbb[nextseq])
                # update end loc
                endseqbb[seqi] = endseqbb[nextseq]
                cseq = nextseq

        if self.RESTRICT_BRANCH_NUM == 0:
            self.seqbbsim = copy.deepcopy(seqbbint)
        else:
            # remove short seq
            seqbbsim = []
            thresremovebranch = 20
            seqmax = np.array([len(seqbbint[i]) for i in range(len(seqbbint))]).argsort()[-pn:][::-1]
            xpos0 = [bbint[seqbbint[seqmax[0]][seqslicei][0]][seqbbint[seqmax[0]][seqslicei][1]].x for seqslicei in
                     range(len(seqbbint[seqmax[0]]))]
            meanx0 = np.mean(xpos0)
            stdx0 = np.std(xpos0)
            xpos1 = [bbint[seqbbint[seqmax[1]][seqslicei][0]][seqbbint[seqmax[1]][seqslicei][1]].x for seqslicei in
                     range(len(seqbbint[seqmax[1]]))]
            meanx1 = np.mean(xpos1)
            stdx1 = np.std(xpos1)
            print('Top tracklets', 'len', len(seqbbint[seqmax[0]]), 'xmean', meanx0, '+-', stdx0, 'len',
                  len(seqbbint[seqmax[1]]), 'xmean', meanx1, '+-', stdx1)
            for seqi in range(len(seqbbint)):
                if len(seqbbint[seqi]) < 3:
                    if DEBUG:
                        print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
                        seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
                        self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                              'with', len(seqbbint[seqi]), 'points')
                    continue
                elif len(seqbbint[seqi]) < 5:
                    sumartp = 0
                    for si in range(len(seqbbint[seqi])):
                        sumartp += bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]].get_score()
                    if sumartp < 0.5 * len(seqbbint[seqi]):
                        if DEBUG:
                            print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
                            seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
                            self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                                  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
                        continue
                    else:
                        if DEBUG:
                            print('keep seqi', seqi, '(%d,%d-%d,%d)' % (
                            seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
                            self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                                  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
                # for longer seqs, stricter rule if not belong to prior defined number of tracklets pn
                elif pn > 0 and seqi not in np.array([len(seqbbint[i]) for i in range(len(seqbbint))]).argsort()[-pn:][
                                            ::-1]:
                    sumartp = 0
                    for si in range(len(seqbbint[seqi])):
                        # if seqbbint[seqi][si][1]<len(bbr[seqbbint[seqi][si][0]]) and \
                        #	bbr[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]]==bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]]:
                        sumartp += bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]].get_score()
                    if sumartp < 0.35 * len(seqbbint[seqi]):
                        print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
                        seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
                        self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                              'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
                        continue
                    else:
                        print('keep seqi', seqi, '(%d,%d-%d,%d)' % (
                        seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
                        self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                              'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))

                if self.usebranch and pn > 0:

                    if seqi not in seqmax:
                        xcenter = []
                        for seqslicei in range(len(seqbbint[seqi])):
                            box = bbint[seqbbint[seqi][seqslicei][0]][seqbbint[seqi][seqslicei][1]]
                            xcenter.append(box.x)
                        meancenter = np.mean(xcenter)
                        mindis0 = abs(meancenter - meanx0)
                        mindis1 = abs(meancenter - meanx1)
                        # if np.mean(xcenter)>330 and np.mean(xcenter)<390:
                        # 	if DEBUG:
                        # 		print(seqi,'(%d,%d-%d,%d)'%(seqbbint[seqi][-1][0]-seqbbint[seqi][0][0],seqbbint[seqi][0][0],seqbbint[seqi][-1][0],self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
                        # 		  'center box,skip')
                        # 	continue
                        print(seqi, 'min dist to main branches', mindis0, mindis1)
                        if mindis0 < 2 * stdx0 or mindis1 < 2 * stdx1:
                            print('Keep')
                        else:
                            continue
                        USE3DDIST = 0
                        # 3d distance to two main branches
                        if USE3DDIST:
                            cbranchbb0s = [seqbbint[seqi][0][0]] + bbint[seqbbint[seqi][0][0]][
                                                                       seqbbint[seqi][0][1]].getbb()[:2]
                            cbranchbb0e = [seqbbint[seqi][-1][0]] + bbint[seqbbint[seqi][-1][0]][
                                                                        seqbbint[seqi][-1][1]].getbb()[:2]

                            mindis = 99999
                            for slicei in range(len(seqbbint[seqmax[0]])):
                                maxbranchbb = [seqbbint[seqmax[0]][slicei][0]] + bbint[seqbbint[seqmax[0]][slicei][0]][
                                                                                     seqbbint[seqmax[0]][slicei][
                                                                                         1]].getbb()[:2]
                                cdists = np.sqrt(
                                    np.sum([(cbranchbb0s[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
                                cdiste = np.sqrt(
                                    np.sum([(cbranchbb0e[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
                                if cdists < mindis:
                                    mindis = cdists
                                if cdiste < mindis:
                                    mindis = cdiste
                            for slicei in range(len(seqbbint[seqmax[1]])):
                                maxbranchbb = [seqbbint[seqmax[1]][slicei][0]] + bbint[seqbbint[seqmax[1]][slicei][0]][
                                                                                     seqbbint[seqmax[1]][slicei][
                                                                                         1]].getbb()[:2]
                                cdists = np.sqrt(
                                    np.sum([(cbranchbb0s[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
                                cdiste = np.sqrt(
                                    np.sum([(cbranchbb0e[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
                                if cdists < mindis:
                                    mindis = cdists
                                if cdiste < mindis:
                                    mindis = cdiste
                            print(seqi, 'min dist to main branches', mindis)
                            if mindis > thresremovebranch:
                                continue
                            else:
                                print('Keep')
                self.seqbbsim.append(copy.deepcopy(seqbbint[seqi]))

        if self.RESTRICT_BRANCH_NUM == 1 and pn == 1:
            # further refine per slice bb number
            self.seqbbsimpn = []
            artsumartp = []
            for seqi in range(len(self.seqbbsim)):
                sumartp = 0
                for si in range(len(self.seqbbsim[seqi])):
                    sumartp += bbint[self.seqbbsim[seqi][si][0]][self.seqbbsim[seqi][si][1]].get_score()
                artsumartp.append(sumartp)
            ordartsum = np.argsort(artsumartp)[::-1]
            bbnumseq = np.zeros((len(bbint)))
            bbnumseqid = np.zeros((len(bbint)), dtype=np.uint8)
            for seqi in range(len(self.seqbbsim)):
                if seqi in ordartsum[:pn]:
                    for se in self.seqbbsim[seqi]:
                        bbnumseq[se[0]] += 1
                        bbnumseqid[se[0]] = se[1]
            if DEBUG:
                print(bbnumseq, bbnumseqid)
            for seqi in range(len(self.seqbbsim)):
                if seqi not in ordartsum[:pn]:
                    # copy with missing bb
                    bbadd = []
                    for se in self.seqbbsim[seqi]:
                        if bbnumseq[se[0]] < pn:
                            # if DEBUG:
                            #	print(se,'extra bb added')
                            nearestbb = None
                            for ofi in range(1, len(bbint)):
                                if se[0] - ofi >= 0 and bbnumseq[se[0] - ofi] > 0:
                                    nearestbb = bbint[int(se[0] - ofi)][bbnumseqid[int(se[0] - ofi)]]
                                    break
                                if se[0] + ofi < len(bbnumseq) and bbnumseq[se[0] + ofi] > 0:
                                    nearestbb = bbint[int(se[0] + ofi)][bbnumseqid[int(se[0] + ofi)]]
                                    break
                            if nearestbb is not None:
                                print(bbint[se[0]][se[1]], 'nearestbb', nearestbb)
                                fitbbnei = bbint[se[0]][se[1]].get_bb_fit(nearestbb)
                                print(se, 'fitbbnei', fitbbnei)
                                if fitbbnei < 20:
                                    bbadd.append(se)
                                    bbnumseq[se[0]] += 1
                                    bbnumseqid[se[0]] = se[1]
                            else:
                                print('No neigh...not possible', se[0])
                    if len(bbadd):
                        self.seqbbsimpn.append(bbadd)
                        if DEBUG:
                            print('add extra bb, len', len(bbadd))
                else:
                    self.seqbbsimpn.append(copy.deepcopy(self.seqbbsim[seqi]))
            self.seqbbsim = self.seqbbsimpn

        self.bbint = bbint
        # update bbs with only valid seqsim
        self.bbs = [[] for i in range(len(bbint))]
        for seqi in range(len(self.seqbbsim)):
            for i in range(len(self.seqbbsim[seqi])):
                slicei = self.seqbbsim[seqi][i][0]
                orislicebbid = self.seqbbsim[seqi][i][1]
                self.seqbbsim[seqi][i] = (slicei, len(self.bbs[slicei]))
                self.bbs[slicei].append(copy.deepcopy(bbint[slicei][orislicebbid]))

        if self.trackletreffigpath is not None:
            self.drawrfseqbb(savename=self.trackletreffigpath + 'ref.png')
        else:
            self.drawrfseqbb()

        # adjust boundary of bb to edge
        if boundaryadjustment:
            self.ba()
            self.interpseqneighgap()
        # remove bb with similar detection
        self.removeduplicatebb()

        self.smoothbb(fig=0)
        if self.trackletreffigpath is not None:
            self.drawrfseqbb(savename=self.trackletreffigpath + 'refba.png')
        else:
            self.drawrfseqbb()

    def smoothbb(self, fig=1):
        seqbbsim = self.seqbbsim
        dim = ['x', 'y', 'w', 'h']
        col = len(dim) + self.classsize
        # smooth trace
        if fig:
            pyplot.figure(figsize=(15, 10))
        for fi in range(len(dim) + self.classsize):
            for seqi in range(len(seqbbsim)):
                if len(seqbbsim[seqi]) < self.medks:
                    # if fi==0:
                    #	print(seqi,'len<',self.medks,', no smooth')
                    continue
                # get all slice id as y axis
                sliceq = [seqbbsim[seqi][i][0] for i in range(len(seqbbsim[seqi]))]
                if fi < 4:
                    yseq = [getattr(self.bbs[seqbbsim[seqi][slicei][0]][seqbbsim[seqi][slicei][1]], dim[fi]) for slicei
                            in range(len(sliceq))]
                else:
                    yseq = [self.bbs[seqbbsim[seqi][slicei][0]][seqbbsim[seqi][slicei][1]].classes[fi - 4] for slicei in
                            range(len(sliceq))]
                yseqsmooth = scipy.signal.medfilt(yseq, kernel_size=self.medks)

                # smooth using mean of neighbor
                yseqsmoothmean = []
                for si in range(len(yseqsmooth)):
                    meany = [yseqsmooth[si]]
                    cslicei = seqbbsim[seqi][si][0]
                    for addi in range(1, self.smoothfilterhsz + 1):
                        if si - addi > 0:
                            slicei = seqbbsim[seqi][si - addi][0]
                            if cslicei - slicei == addi:
                                meany.append(yseqsmooth[si - addi])
                        if si + addi < len(yseqsmooth):
                            slicei = seqbbsim[seqi][si + addi][0]
                            if slicei - cslicei == addi:
                                meany.append(yseqsmooth[si + addi])
                    yseqsmoothmean.append(np.mean(meany))
                yseqsmooth = yseqsmoothmean
                # yseqsmooth = [yseqsmooth[0]]+[np.mean(yseqsmooth[i-1:i+2]) for i in range(1,len(yseqsmooth)-1)]+[yseqsmooth[-1]]
                if fig:
                    ax1 = pyplot.subplot(2, col, fi + 1)
                    ax1.plot(yseq, sliceq, 'o', label='Trace%d' % seqi)
                    # ax1.set_ylim(483.3, 344.7)
                    ax2 = pyplot.subplot(2, col, col + fi + 1)
                    ax2.plot(yseqsmooth, sliceq, 'o', label='Trace%d' % seqi)
                # record in bbs
                for i in range(len(sliceq)):
                    slicei = sliceq[i]
                    slicebbid = seqbbsim[seqi][i][1]
                    if fi < 4:
                        setattr(self.bbs[slicei][slicebbid], dim[fi], yseqsmooth[i])
                    else:
                        self.bbs[slicei][slicebbid].classes[fi - 4] = yseqsmooth[i]

            if fig:
                # pyplot.legend(loc="lower right")
                if fi < 4:
                    ax1.set_title(dim[fi] + " Before")
                    ax2.set_title(dim[fi] + " After")
                else:
                    ax1.set_title('class %d Before' % (fi - 4))
                    ax2.set_title('class %d After' % (fi - 4))
                # pyplot.xlim([200,600])
                ax1.set_ylim(max(ax1.get_ylim()), min(ax1.get_ylim()))
                ax2.set_ylim(max(ax1.get_ylim()), min(ax1.get_ylim()))
        # pyplot.gca().invert_yaxis()

        if fig:
            pyplot.show()

    def setresolution(self, res):
        self.RESOLUTION = res

    def settrackletreffigpath(self, trackletreffigpath):
        self.trackletreffigpath = trackletreffigpath

    def ba(self):
        # boundary adjustment
        DEBUG = 0
        if self.imgstack is None:
            print('No image stack, skip boundary adjustment')
            return
        if DEBUG:
            # x y positions for 3 situations(before, after baref, and after smooth)
            xypos = np.zeros((len(self.bbs), 3, 2))
        NEEDSEG = 0
        if self.gwsegstack is None:
            NEEDSEG = 1
            self.gwsegstack = np.zeros((self.imgstack.shape))
        else:
            print('use seg for ba')
        # use great wall seg for boundary info
        gwmodel = self.gwmodel
        assert self.gwmodel is not None
        seqbbsim = self.seqbbsim
        seqbbsimori = copy.deepcopy(seqbbsim)

        for seqi in range(len(seqbbsim)):
            for seqslicei in range(len(seqbbsim[seqi]) - 1, -1, -1):
                if seqslicei % 5 == 0:
                    print('\rBA', seqi, '/', len(seqbbsim), seqslicei, '/', len(seqbbsim[seqi]), end="")
                slicei = seqbbsim[seqi][seqslicei][0]
                bbi = seqbbsim[seqi][seqslicei][1]
                cbb = self.bbs[slicei][bbi]
                rx = int(round(cbb.x))
                ry = int(round(cbb.y))
                if self.RESOLUTION == -1:
                    cpsz = 32
                else:
                    cpsz = int(np.ceil(32 / (self.RESOLUTION / gwmodel.gwmodelRes)))
                if NEEDSEG == 0:
                    probpatchnorm = croppatch(self.gwsegstack[slicei], cbb.y, cbb.x, cpsz, cpsz)
                    probpatch = probpatchnorm / 255
                else:
                    dcmpatch = croppatch(self.imgstack[slicei], cbb.y, cbb.x, cpsz, cpsz)
                    if np.max(dcmpatch) != 0:
                        dcmpatchnorm = dcmpatch / np.max(dcmpatch)
                    else:
                        print('max pixel zeros at cropped patch')
                        continue

                    # use great wall segmentation to generate a rough prob map for center adjustment
                    if self.RESOLUTION == -1:
                        dcmres = -1
                    else:
                        dcmres = self.RESOLUTION
                    probpatch = gwmodel.predprob(dcmpatchnorm, dcmres)
                    probpatchnorm = probpatch / np.max(probpatch) * 255
                    # fill vessel image stack
                    self.gwsegstack[slicei] = fillpatch(self.gwsegstack[slicei], probpatchnorm, ry, rx)
                # self.gwsegstack[slicei,ry-halfpatchsize:ry+halfpatchsize,rx-halfpatchsize:rx+halfpatchsize] = probpatchnorm
                halfpatchsize = probpatchnorm.shape[0] // 2
                if DEBUG:
                    # paint before and after adjustment
                    xypos[slicei, 0] = self.bbs[slicei][bbi].getbb()[0:2]
                    pyplot.figure(figsize=(10, 10))
                    cbblist = [int(xypos[slicei][0][i]) for i in range(2)]
                    probimgpatch = croppatch(self.imgstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
                                             halfpatchsize)
                    probimgpatch[halfpatchsize, halfpatchsize - 1:halfpatchsize + 2] = 255
                    probimgpatch[halfpatchsize - 1:halfpatchsize + 2, halfpatchsize] = 255
                    pyplot.subplot(2, 2, 1)
                    pyplot.title('prob ori')
                    pyplot.imshow(probimgpatch)
                    segimgpatch = croppatch(self.gwsegstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
                                            halfpatchsize)
                    pyplot.subplot(2, 2, 2)
                    pyplot.title('prob ori seg')
                    pyplot.imshow(segimgpatch)

                # predict boundary adjustment
                adjlist = self._refct2(probpatch)
                if adjlist is None:
                    # print('remove',seqi,seqslicei,slicei,rx)
                    # pyplot.show()
                    del seqbbsim[seqi][seqslicei]
                    continue
                if DEBUG:
                    print(slicei, adjlist)
                self.bbs[slicei][bbi].x = rx + adjlist[0]
                self.bbs[slicei][bbi].y = ry + adjlist[1]
                if DEBUG:
                    xypos[slicei, 1] = self.bbs[slicei][bbi].getbb()[0:2]
                    cbblist = [int(xypos[slicei][1][i]) for i in range(2)]
                    probimgpatch = croppatch(self.imgstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
                                             halfpatchsize)
                    probimgpatch[halfpatchsize, halfpatchsize - 1:halfpatchsize + 2] = 255
                    probimgpatch[halfpatchsize - 1:halfpatchsize + 2, halfpatchsize] = 255
                    pyplot.subplot(2, 2, 3)
                    pyplot.title('prob refct %d %d' % (
                    xypos[slicei][1][0] - xypos[slicei][0][0], xypos[slicei][1][1] - xypos[slicei][0][1]))
                    pyplot.imshow(probimgpatch)
                    segimgpatch = croppatch(self.gwsegstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
                                            halfpatchsize)
                    pyplot.subplot(2, 2, 4)
                    pyplot.title('prob refct seg')
                    pyplot.imshow(segimgpatch)
                    pyplot.show()

            if len(seqbbsim[seqi]) > 100:
                print('seq', seqi, 'main branch, no remove')
                seqbbsim[seqi] = seqbbsimori[seqi]

        # smooth again after center adjustment
        if DEBUG:
            self.smoothbb(fig=1)
        else:
            self.smoothbb(fig=0)

    # assume only one bounding box per slice
    # if DEBUG:
    # 	for slicei in range(len(self.bbs)):
    # 		xypos[slicei,2] = self.bbs[slicei][0].getbb()[0:2]
    # if DEBUG:
    # 	#display difference of pos before after ref ct
    # 	for slicei in range(len(self.bbs)):
    # 		pyplot.figure(figsize=(10,4))
    # 		cbblist = [int(xypos[slicei][0][i]) for i in range(2)]
    # 		print(slicei,cbblist)
    # 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
    # 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
    # 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
    # 		pyplot.subplot(1,3,1)
    # 		pyplot.title('prob ori')
    # 		pyplot.imshow(probimgpatch)

    # 		cbblist = [int(xypos[slicei][1][i]) for i in range(2)]
    # 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
    # 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
    # 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
    # 		pyplot.subplot(1,3,2)
    # 		pyplot.title('prob refct %d %d'%(xypos[slicei][1][0]-xypos[slicei][0][0],xypos[slicei][1][1]-xypos[slicei][0][1]))
    # 		pyplot.imshow(probimgpatch)

    # 		cbblist = [int(xypos[slicei][2][i]) for i in range(2)]
    # 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
    # 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
    # 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
    # 		pyplot.subplot(1,3,3)
    # 		pyplot.title('prob smooth %d %d'%(xypos[slicei][2][0]-xypos[slicei][0][0],xypos[slicei][2][1]-xypos[slicei][0][1]))
    # 		pyplot.imshow(probimgpatch)
    # 		pyplot.show()

    def _refct2(self, probimg):
        maxtl = 0
        gap = 3
        mcy = probimg.shape[0] // 2
        mcx = probimg.shape[1] // 2

        nd = scipy.stats.norm(13, 5)
        pdf = [nd.pdf(i) for i in range(int(probimg.shape[0] * 1.5))]
        for i in range(13):
            pdf[i] = pdf[13]

        for yi in range(2, probimg.shape[0] - 2, gap):
            for xi in range(2, probimg.shape[1] - 2, gap):
                cx1 = probimg[yi::-1, xi]
                cx2 = probimg[yi:, xi]
                cy1 = probimg[yi, xi::-1]
                cy2 = probimg[yi, xi:]
                # RB
                cxy1 = [probimg[yi + i, xi + i] for i in range(min(probimg.shape[0] - yi, probimg.shape[1] - xi))]
                # RU
                cxy2 = [probimg[yi - i, xi + i] for i in range(min(yi, probimg.shape[1] - xi))]
                # LB
                cxy3 = [probimg[yi + i, xi - i] for i in range(min(probimg.shape[0] - yi, xi))]
                # LU
                cxy4 = [probimg[yi - i, xi - i] for i in range(min(yi, xi))]
                ctint = probimg[yi, xi]
                lcx1 = np.max([i * abs(cx1[i] - cx1[i - 1]) * pdf[i] for i in range(1, len(cx1))])
                lcx2 = np.max([i * abs(cx2[i] - cx2[i - 1]) * pdf[i] for i in range(1, len(cx2))])
                lcy1 = np.max([i * abs(cy1[i] - cy1[i - 1]) * pdf[i] for i in range(1, len(cy1))])
                lcy2 = np.max([i * abs(cy2[i] - cy2[i - 1]) * pdf[i] for i in range(1, len(cy2))])
                lcxy1 = np.max([abs(cxy1[i] - cxy1[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy1))])
                lcxy2 = np.max([abs(cxy2[i] - cxy2[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy2))])
                lcxy3 = np.max([abs(cxy3[i] - cxy3[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy3))])
                lcxy4 = np.max([abs(cxy4[i] - cxy4[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy4))])
                tl = lcx1 * lcx2 * lcy1 * lcy2 * lcxy1 * lcxy2 * lcxy3 * lcxy4 * (1 - ctint)
                if tl > maxtl:
                    # print(xi,yi,tl)
                    maxtl = tl
                    mcx = xi
                    mcy = yi
        # print('maxtl',maxtl)
        if maxtl < 1e-10:
            # print('maxtl',maxtl)
            return None
        else:
            return [mcx - probimg.shape[1] // 2, mcy - probimg.shape[0] // 2]

    def removeduplicatebb(self):
        # remove detection overlaps
        # start from longest seq, remove bb from other seq if overlap
        argseqlen = np.argsort([len(seqi) for seqi in self.seqbbsim])[::-1]
        for seqidx in range(len(argseqlen)):
            seqid = argseqlen[seqidx]
            seqsegconfs = []
            for seqbbid in range(len(self.seqbbsim[seqid]) - 1, -1, -1):
                slicei = self.seqbbsim[seqid][seqbbid][0]
                bbid = self.seqbbsim[seqid][seqbbid][1]
                bbi = self.bbs[slicei][bbid]
                # search other shorter seq
                for seqjdx in range(seqidx + 1, len(argseqlen)):
                    seqjd = argseqlen[seqjdx]
                    for seqbbjd in range(len(self.seqbbsim[seqjd]) - 1, -1, -1):
                        if slicei == self.seqbbsim[seqjd][seqbbjd][0]:
                            bbjd = self.seqbbsim[seqjd][seqbbjd][1]
                            bbj = self.bbs[slicei][bbjd]
                            bbiom = bbi.get_iom(bbj)
                            if bbiom > 0.5:
                                del self.seqbbsim[seqjd][seqbbjd]
                        # print('remove duplicate bb',slicei,seqid,seqbbid,seqjd,seqbbjd)
        # update bbs after removing some items in seqbb
        self.refreshseq()

    def refreshseq(self):
        if self.bbs is None:
            print('no bbs')
            return
        bbsnew = [[] for i in range(len(self.bbs))]
        seqbbsim = copy.deepcopy(self.seqbbsim)
        for seqid in range(len(seqbbsim)):
            for seqbbid in range(len(seqbbsim[seqid])):
                slicei = seqbbsim[seqid][seqbbid][0]
                bbid = seqbbsim[seqid][seqbbid][1]
                seqbbsim[seqid][seqbbid] = (slicei, len(bbsnew[slicei]))
                bbsnew[slicei].append(copy.deepcopy(self.bbs[slicei][bbid]))
        self.bbs = bbsnew
        self.seqbbsim = seqbbsim

    def interpseqneighgap(self):
        # interpolate the gap casued by ba remove bb within each seqbbsim
        # if there are neighboring bbs within 5 slices, use mean of all neigbors as the result
        newseqbb = [[] for i in range(len(self.seqbbsim))]
        for seqid in range(len(self.seqbbsim)):
            if len(self.seqbbsim[seqid]) == 0:
                continue
            startslicei = self.seqbbsim[seqid][0][0]
            endslicei = self.seqbbsim[seqid][-1][0]
            for slicei in range(startslicei, endslicei + 1):
                # search existing seqbbsim
                fd = -1
                bbneiitems = []
                for seqbbid in range(len(self.seqbbsim[seqid])):
                    slicej = self.seqbbsim[seqid][seqbbid][0]
                    if slicej == slicei:
                        newseqbb[seqid].append(self.seqbbsim[seqid][seqbbid])
                        fd = slicej
                        break
                    if abs(slicei - slicej) < 5:
                        sliceni = self.seqbbsim[seqid][seqbbid][0]
                        bbnei = self.seqbbsim[seqid][seqbbid][1]
                        bbitem = self.bbs[sliceni][bbnei].getbbclassflat()
                        bbneiitems.append(bbitem)

                if fd != -1:
                    continue
                # if not found, inter if there are neighbor bb
                elif fd == -1 and len(bbneiitems):
                    meanbbitem = []
                    for fi in range(len(bbneiitems[0])):
                        meanbbitem.append(np.mean([bbneiitems[bbni][fi] for bbni in range(len(bbneiitems))]))
                    newseqbb[seqid].append((slicei, len(self.bbs[slicei])))
                    self.bbs[slicei].append(BB.fromlistclass(meanbbitem))
        # else:
        # print(seqid,slicei,'gap interp')
        self.seqbbsim = newseqbb

    def getbbs(self, clabel=0, minmax=0):
        if self.bbs is None:
            print('Refine first')
            return
        bbs = [[] for i in range(len(self.bbs))]
        for slicei in range(len(self.bbs)):
            bbsslice = []
            for bbi in range(len(self.bbs[slicei])):
                if clabel == 0:
                    if minmax == 0:
                        bbsslice.append(self.bbs[slicei][bbi].getbb())
                    else:
                        bbsslice.append(self.bbs[slicei][bbi].getminmax())
                else:
                    if minmax == 0:
                        bbsslice.append(self.bbs[slicei][bbi].getbbclabel())
                    else:
                        bbsslice.append(self.bbs[slicei][bbi].getminmaxclabel())
            if len(bbsslice):
                bbs[slicei] = bbsslice
        return bbs

    def getbbsclass(self):
        if self.bbs is None:
            print('Refine first')
            return
        bbs = [[] for i in range(len(self.bbs))]
        for slicei in range(len(self.bbs)):
            bbsslice = []
            for bbi in range(len(self.bbs[slicei])):
                bbsslice.extend(self.bbs[slicei][bbi].getbbclass())
            if len(bbsslice):
                if bbsslice[0] == 0:
                    print('bbsslice 0')
                    continue
                bbs[slicei].append(bbsslice)
        return bbs

    # def fillmissing(self):
    # 	if self.bbs is None:
    # 		print('Refine first')
    # 		return
    # 	bbs=[[] for i in range(len(self.bbs))]
    # 	for slicei in range(len(self.bbs)):

    # write refined boxes to bbrfdir
    def writebbs(self, bbrffolder, einame):
        if not os.path.exists(bbrffolder):
            os.mkdir(bbrffolder)
        else:
            for fi in os.listdir(bbrffolder):
                os.remove(bbrffolder + '/' + fi)
            print('remove existing bbs')
        bbs = self.bbs
        for slicei in range(len(bbs)):
            slicename = os.path.join(bbrffolder, einame + '_%03d' % slicei + '.txt')
            with open(slicename, 'w') as file:
                file.write('%d\n' % len(bbs[slicei]))
                for bbiBB in bbs[slicei]:
                    bbi = bbiBB.getminmaxclabel()
                    # print(slicei,box)
                    # label tool format
                    file.write('%.2f %.2f %.2f %.2f %.2f %d\n' % (bbi[0], bbi[1], bbi[2], bbi[3], bbi[4], bbi[5]))

    def writebbr(self, bbfolder, einame):
        if not os.path.exists(bbfolder):
            os.mkdir(bbfolder)
        else:
            print('remove existing bbr', os.listdir(bbfolder))
            for fi in os.listdir(bbfolder):
                os.remove(bbfolder + '/' + fi)

        bbr = self.bbr
        for slicei in range(len(bbr)):
            slicename = os.path.join(bbfolder, einame + '_%03d' % slicei + '.txt')
            with open(slicename, 'w') as file:
                file.write('%d\n' % len(bbr[slicei]))
                for bbiBB in bbr[slicei]:
                    bbi = bbiBB.getminmaxclabel()
                    # print(slicei,box)
                    # label tool format
                    file.write('%.2f %.2f %.2f %.2f %.2f %d\n' % (bbi[0], bbi[1], bbi[2], bbi[3], bbi[4], bbi[5]))

    # connect bb using minimum intensity path
    def connectbb(self, featuremodel=None, config=None):
        DEBUG = 1
        lossthres = 1
        if config is None:
            pass
        else:
            if 'debug' in config:
                DEBUG = config['debug']
        if DEBUG:
            self.drawseqbb()
        self.seqbbsim = []
        seqbb = copy.deepcopy(self.seqbb)
        # starting/ending slice id of each seq
        startseqbb = [seqbb[i][0][0] for i in range(len(seqbb))]
        endseqbb = [seqbb[i][-1][0] for i in range(len(seqbb))]

        connectend = []
        connectstart = []
        # find seqbb after end of each seqbb
        for seqi in range(len(seqbb)):
            # skip if already end of slice
            if endseqbb[seqi] == len(self.bbr):
                continue

            # already has connection
            if seqi in connectend:
                continue
            if DEBUG:
                print('seqid', seqi, 'slicei', seqbb[seqi][0][0], '-', seqbb[seqi][-1][0], 'bb start',
                      self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].getbb())

            cminloss = np.inf
            seqminind = -1

            availseqj = [i for i in range(len(startseqbb)) if
                         startseqbb[i] >= endseqbb[seqi] and i not in connectstart + [seqi]]
            if DEBUG:
                print('avail seqj', availseqj)
            for seqj in availseqj:
                # find seqi end bb
                slicetuplei = seqbb[seqi][-1]
                bbi = copy.deepcopy(self.bbr[slicetuplei[0]][slicetuplei[1]])
                # find seqj start bb
                slicetuplej = seqbb[seqj][0]
                bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
                # calculate loss
                cgap = startseqbb[seqj] - endseqbb[seqi]
                diffx = abs(bbi.x - bbj.x)
                diffy = abs(bbi.y - bbj.y)
                if (diffx + diffy > 50) or cgap > 25:
                    # print(seqi,seqj, 'dif',diffx,diffy)
                    continue

                pos1 = [bbi.x, bbi.y, endseqbb[seqi]]
                pos2 = [bbj.x, bbj.y, startseqbb[seqj]]
                if featuremodel is None:
                    intensity_along_path = np.array(intpath(pos1, pos2, self.imgstack))
                    intensity_along_path_d = abs(intensity_along_path[1:] - intensity_along_path[:-1])
                    closs = np.sum(intensity_along_path_d)
                else:
                    if DEBUG:
                        print(pos1, pos2)
                    closs = bb_match_score(featuremodel, self.imgstack, pos1, pos2) / 20
                if DEBUG:
                    print(seqi, '(%d,%d-%d,%d)' % (
                        seqbb[seqi][-1][0] - seqbb[seqi][0][0], seqbb[seqi][0][0], seqbb[seqi][-1][0],
                        self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].x),
                          seqj, '(%d,%d-%d,%d)' % (
                              seqbb[seqj][-1][0] - seqbb[seqj][0][0], seqbb[seqj][0][0], seqbb[seqj][-1][0],
                              self.bbr[seqbb[seqj][0][0]][seqbb[seqj][0][1]].x),
                          cgap, closs)
                if closs < lossthres:
                    if closs < cminloss:
                        cminloss = closs
                        seqminind = seqj

            if seqminind == -1:
                # print('all start of seq not fit')
                continue
            else:
                # print('End of ',seqi,' to start of ',seqminind,' has min loss, test other ends')
                slicetuplej = seqbb[seqminind][0]
                bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
                ciminloss = np.inf
                seqiminind = -1
                availseqii = [i for i in range(len(endseqbb)) if
                              endseqbb[i] <= startseqbb[seqminind] and i not in connectend + [seqminind]]
                if DEBUG:
                    print('availseqii', availseqii)
                for seqii in availseqii:
                    # find seqii end bb
                    slicetupleii = seqbb[seqii][-1]
                    bbii = copy.deepcopy(self.bbr[slicetupleii[0]][slicetupleii[1]])
                    # calculate loss
                    cgap = startseqbb[seqminind] - endseqbb[seqii]
                    diffx = abs(bbii.x - bbj.x)
                    diffy = abs(bbii.y - bbj.y)
                    if diffx + diffy > 50 or cgap > 25:
                        continue

                    pos1 = [bbii.x, bbii.y, endseqbb[seqii]]
                    pos2 = [bbj.x, bbj.y, startseqbb[seqminind]]
                    if featuremodel is None:
                        intensity_along_path = np.array(intpath(pos1, pos2, self.imgstack))
                        intensity_along_path_d = abs(intensity_along_path[1:] - intensity_along_path[:-1])
                        closs = np.sum(intensity_along_path_d)
                    else:
                        if DEBUG:
                            print(pos1, pos2)
                        closs = bb_match_score(featuremodel, self.imgstack, pos1, pos2) / 20
                    if DEBUG:
                        print('R', seqii, seqminind, cgap, closs)
                    if closs < lossthres:
                        if closs < ciminloss:
                            ciminloss = closs
                            seqiminind = seqii
                if seqiminind == seqi:
                    if DEBUG:
                        print('back match, connect', seqi, seqminind)
                    # connect from i to seqminind
                    connectend.append(seqi)
                    connectstart.append(seqminind)
                else:
                    if DEBUG:
                        print('back mismatch', seqi, seqiminind, seqminind)

        # deep copy
        bbint = copy.deepcopy(self.bbr)
        seqbbint = []
        # make connection with start not in connect start
        for seqi in range(len(seqbb)):
            if seqi in connectstart:
                continue
            seqbbint.append(copy.deepcopy(seqbb[seqi]))
            cseq = seqi
            while cseq in connectend:
                nextseq = connectstart[connectend.index(cseq)]
                if DEBUG:
                    print('connect', cseq, nextseq)
                bbend = bbint[seqbb[cseq][-1][0]][seqbb[cseq][-1][1]].getbbclassflat()
                bbstart = self.bbr[seqbb[nextseq][0][0]][seqbb[nextseq][0][1]].getbbclassflat()
                if DEBUG:
                    print('bbstart', bbstart, 'end', bbend)
                bbdiff = [bbstart[i] - bbend[i] for i in range(len(bbend))]
                gap = startseqbb[nextseq] - endseqbb[seqi]
                if DEBUG:
                    print('gap', gap, 'diff', bbdiff)
                for coni in range(endseqbb[seqi] + 1, startseqbb[nextseq]):
                    if DEBUG:
                        print('appending slice', coni)
                        print([bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))])
                    seqbbint[-1].append((coni, len(bbint[coni])))
                    intpbblist = [bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))]
                    bbint[coni].append(BB.fromlistclass(intpbblist))
                    if DEBUG:
                        print(len(seqbb[0]), 'cseq')
                seqbbint[-1].extend(seqbb[nextseq])
                # update end loc
                endseqbb[seqi] = endseqbb[nextseq]
                cseq = nextseq

        self.seqbbsim = copy.deepcopy(seqbbint)

        self.bbint = bbint
        # update bbs with only valid seqsim
        self.bbs = [[] for i in range(len(bbint))]
        for seqi in range(len(self.seqbbsim)):
            for i in range(len(self.seqbbsim[seqi])):
                slicei = self.seqbbsim[seqi][i][0]
                orislicebbid = self.seqbbsim[seqi][i][1]
                self.seqbbsim[seqi][i] = (slicei, len(self.bbs[slicei]))
                self.bbs[slicei].append(copy.deepcopy(bbint[slicei][orislicebbid]))

        # if self.trackletreffigpath is not None:
        #	self.drawrfseqbb(savename=self.trackletreffigpath + 'ref.png')
        if DEBUG:
            self.drawrfseqbb()

    def seqbbsim_stat(self):
        stats = []
        for seq in self.seqbbsim:
            xpos = []
            ypos = []
            cpos = []
            for seqi in seq:
                slicei = seqi[0]
                bbid = seqi[1]
                bb = self.bbs[slicei][bbid]
                xpos.append(bb.x)
                ypos.append(bb.y)
                cpos.append(bb.c)
            stats.append([np.mean(xpos), np.mean(ypos), np.mean(cpos), len(seq)])
        return np.array(stats)

    def label_carotid_art(self):
        seqstat = np.array(self.seqbbsim_stat())
        seqlen = seqstat[:, 2]
        arglen = np.argsort(seqlen)[::-1]

    def calseqproperty(self, neckxcenter=256):
        ARTTYPE = ['ICAL', 'ECAL', 'ICAR', 'ECAR']
        bbs = self.bbs
        seqbbsim = self.seqbbsim
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

            # if more than 2 on one side
            if len(Lobj) > 2:
                Lconfs = []
                for ai in range(len(Lobj)):
                    Lconfs.append(confs[Lobj[ai]])
                arglconf = np.argsort(Lconfs)[::-1]
                Lobj = [Lobj[arglconf[0]]] + [Lobj[arglconf[1]]]
                print('Multiple bb on left side, top2', Lobj)
            if len(Robj) > 2:
                Rconfs = []
                for ai in range(len(Robj)):
                    Rconfs.append(confs[Robj[ai]])
                print(Rconfs)
                argrconf = np.argsort(Rconfs)[::-1]
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
        # print(bblabel)

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

    def seqbbsim_iou_tracklet(self, seqid, bbcomp):
        seq = self.seqbbsim[seqid]
        bbiou = []
        for seqi in seq:
            slicei = seqi[0]
            bbid = seqi[1]
            bb = self.bbs[slicei][bbid]
            if len(bbcomp[slicei]) > 0:
                bbc = bbcomp[slicei][0]
            else:
                print('no bbc on slice', slicei)
                continue
            bbiou.append(bb.get_iou(bbc))
            print(slicei, bbiou[-1])
        return np.mean(bbiou)

    # get tracklet size of num from longest seqbbsim. if num on certain slice reaches, ignore other seqbbsim
    # c: 0-x 1-y 2-conf 3-length
    def collect_tracklet_by_c(self, num=1, c=2):
        Tbbstat = self.seqbbsim_stat()
        tracklet = [[] for i in range(len(self.bbs))]
        tbb_len_sort = np.argsort(Tbbstat[:, c])[::-1]
        print('mean track conf', Tbbstat[:, c], 'sort', tbb_len_sort)
        for seqid in tbb_len_sort:
            for seqi in self.seqbbsim[seqid]:
                slicei = seqi[0]
                if len(tracklet[slicei]) >= num:
                    continue
                bbid = seqi[1]
                bb = self.bbs[slicei][bbid]
                tracklet[slicei].append(copy.copy(bb))
        return tracklet


import copy
import matplotlib.pyplot as pyplot
import numpy as np
import scipy
from scipy import signal
import os
from collections import Counter

from PolarVW.BB import BB
from PolarVW.UTL import croppatch, fillpatch
from PolarVW.triplet import bb_match_score


def intpath(pos1, pos2, dicomstack):
	DEBUG = 0
	pos1 = np.array(pos1)
	pos2 = np.array(pos2)
	pos1int = [int(round(posi)) for posi in pos1]

	pos2int = [int(round(posi)) for posi in pos2]
	direction = pos2 - pos1
	dist = np.linalg.norm(direction)
	dirnorm = direction / dist
	intp = []
	intp.append(dicomstack[pos1int[2]][pos1int[1]][pos1int[0]])
	if DEBUG:
		pdsp = croppatch(dicomstack[pos1int[2]], pos1int[1], pos1int[0])
		pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
		pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
		pyplot.title('start')
		pyplot.imshow(pdsp)
		pyplot.show()

	for stepi in range(1, int(np.floor(dist))):
		cpos = pos1 + dirnorm * stepi
		cposint = np.array([int(np.round(cposi)) for cposi in cpos])
		if DEBUG:
			pdsp = croppatch(dicomstack[cposint[2]], cpos[1], cpos[0])
			pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
			pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
			pyplot.imshow(pdsp)
			pyplot.title(str(cposint[2]))
			pyplot.show()

		valceil = dicomstack[int(np.ceil(cpos[2]))][cposint[1]][cposint[0]]
		valfloor = dicomstack[int(np.floor(cpos[2]))][cposint[1]][cposint[0]]
		w1 = int(np.ceil(cpos[2])) - cpos[2]
		w2 = cpos[2] - int(np.floor(cpos[2]))
		intp.append(valceil * w2 / (w1 + w2) + valfloor * w1 / (w1 + w2))

	intp.append(dicomstack[pos2int[2]][pos2int[1]][pos2int[0]])
	if DEBUG:
		pdsp = croppatch(dicomstack[pos2int[2]], pos2int[1], pos2int[0])
		pdsp[pdsp.shape[0] // 2] = np.max(pdsp)
		pdsp[:, pdsp.shape[1] // 2] = np.max(pdsp)
		pyplot.title('end')
		pyplot.imshow(pdsp)
		pyplot.show()
	return intp


class Tracklet:
	# self.seqbb and self.bbr store tracklet info
	# self.seqbbsim and self.bbs after refinement
	def __init__(self, bbr, classsize, imgstack=None, iouthres=0.65):
		self.RESOLUTION = -1
		self.iouexpandratio = 1.0
		self.imgstack = imgstack
		assert imgstack.shape[0] == len(bbr)
		self.bbr = bbr
		self.classsize = classsize
		# self.bbr = [[] for i in range(len(bbr))]
		# for slicei in range(len(bbr)):
		# 	for bi in range(len(bbr[slicei])):
		# 		x,y,w,h = [float(bbr[slicei][bi][i]) for i in range(4)]
		# 		classes = bbr[slicei][bi][4]

		# 		cbblist = BB(x,y,w,h,classes=classes)
		# 		self.bbr[slicei].append(cbblist)
		self.genseq(iouthres)
		self.bbs = None
		self.seqbbsim = None
		self.trackletreffigpath = None
		self.gwsegstack = None
		self.RESTRICT_BRANCH_NUM = 1

	def __repr__(self):
		rtstr = 'Tracklet seqs:%d' % (len(self.seqbb))
		if self.bbs is None:
			rtstr += ', unrefined'
		else:
			rtstr += ', refined to %d' % (len(self.seqbbsim))
		if self.trackletreffigpath is not None:
			rtstr += ', reffigpath: ' + self.trackletreffigpath
		return rtstr

	def genseq(self, iouthres):
		# continuous sequence, each seq a list with each pt recorded as sliceid, bb id on that slice
		self.seqbb = []
		lbb = []  # last slice bounding box
		lbbid = []  # last slice bb id
		bifseqid = []
		for slicei in range(len(self.bbr)):
			# set available bb in this slice
			availbbslicei = [1 for i in range(len(self.bbr[slicei]))]
			# check last slice whether has continous seq
			lbbi = 0
			while lbbi < len(lbb):
				# find max iou for last slice bb
				cmaxiou = 0
				cmaxbbi = -1
				for bbi in range(len(self.bbr[slicei])):
					ioum = self.bbr[slicei][bbi].get_expand_iou(lbb[lbbi], self.iouexpandratio)
					if ioum > cmaxiou:
						cmaxiou = ioum
						cmaxbbi = bbi
				if cmaxiou < iouthres:
					# not found, stop seq
					del lbb[lbbi]
					del lbbid[lbbi]
				else:
					# print(cmaxiou)
					# update seqbb
					self.seqbb[lbbid[lbbi]].append((slicei, cmaxbbi))
					if availbbslicei[cmaxbbi] == 1:
						availbbslicei[cmaxbbi] = 0
						lbb[lbbi] = copy.deepcopy(self.bbr[slicei][cmaxbbi])
						lbbi += 1
					else:
						# print('bif')
						bifseqid.append(copy.deepcopy(lbbid[lbbi]))
						del lbb[lbbi]
						del lbbid[lbbi]

			# remaining bb set as new lbb
			for avi in range(len(availbbslicei)):
				if availbbslicei[avi] == 1:
					# add as new lbb
					lbb.append(copy.deepcopy(self.bbr[slicei][avi]))
					lbbid.append(len(self.seqbb))
					self.seqbb.append([(slicei, avi)])
		print('seq length', len(self.seqbb))

	def drawseqbb(self, dim='x', savename=None):
		self.connectbb_seqdraw(self.seqbb, self.bbr, dim, savename)

	def drawrfseqbb(self, dim='x', savename=None):
		if self.seqbbsim is None:
			print('Refine first')
			return
		self.connectbb_seqdraw(self.seqbbsim, self.bbs, dim, savename)

	def connectbb_seqdraw(self, seqbb, bb, dim, savename=None):
		for seqi in range(len(seqbb)):
			# print(seqi,len(seqbb[seqi]))
			sliceq = [seqbb[seqi][slicei][0] for slicei in range(len(seqbb[seqi]))]
			xseq = [getattr(bb[seqbb[seqi][slicei][0]][seqbb[seqi][slicei][1]], dim) for slicei in range(len(sliceq))]
			pyplot.plot(xseq, sliceq, 'o', label='Tracklet%d' % seqi)
		# pyplot.legend(loc="lower right")
		pyplot.legend()
		# pyplot.xlim([200,550])
		pyplot.ylabel('slice numbers', fontsize=18)
		pyplot.xlabel(dim + ' position', fontsize=18)
		pyplot.gca().invert_yaxis()
		if savename is not None:
			pyplot.savefig(savename)
		pyplot.show()

	def refbb(self, config=None):
		# pn: prior knowledge for fixed number of tracklets
		DEBUG = 0
		if config is None:
			pn = 1
			boundaryadjustment = 0
		else:
			if 'pn' in config:
				pn = config['pn']
			else:
				pn = 1
			if 'ba' in config:
				boundaryadjustment = config['ba']
			else:
				boundaryadjustment = 0
			if 'medks' in config:
				self.medks = config['medks']
			else:
				self.medks = 11
			if 'smoothfilterhsz' in config:
				self.smoothfilterhsz = config['smoothfilterhsz']
			else:
				self.smoothfilterhsz = 2
			if 'usebranch' in config:
				self.usebranch = config['usebranch']
			else:
				self.usebranch = 1
			if 'debug' in config:
				DEBUG = config['debug']

		self.seqbbsim = []
		seqbb = copy.deepcopy(self.seqbb)
		# starting/ending slice id of each seq
		startseqbb = [seqbb[i][0][0] for i in range(len(seqbb))]
		endseqbb = [seqbb[i][-1][0] for i in range(len(seqbb))]
		# [(i,startseqbb[i],endseqbb[i]) for i in range(len(seqbb))]

		weightiou = 0.5
		weightgap = 0.01
		weightbbfit = 0.001
		lossthres = 0.5
		connectend = []
		connectstart = []
		# find seqbb after end of each seqbb
		for seqi in range(len(seqbb)):
			# skip if already end of slice
			if endseqbb[seqi] == len(self.bbr):
				continue

			# already has connection
			if seqi in connectend:
				continue
			if DEBUG:
				print(seqi, self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].getbb())
				print(seqi, self.bbr[seqbb[seqi][-1][0]][seqbb[seqi][-1][1]].getbb())

			cminloss = np.inf
			seqminind = -1

			availseqj = [i for i in range(len(startseqbb)) if
						 startseqbb[i] >= endseqbb[seqi] and i not in connectstart + [seqi]]
			for seqj in availseqj:
				# find seqi end bb
				slicetuplei = seqbb[seqi][-1]
				bbi = copy.deepcopy(self.bbr[slicetuplei[0]][slicetuplei[1]])
				# find seqj start bb
				slicetuplej = seqbb[seqj][0]
				bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
				# calculate loss
				ciou = bbi.get_expand_iou(bbj, self.iouexpandratio)
				cgap = startseqbb[seqj] - endseqbb[seqi]
				diffx = abs(bbi.x - bbj.x)
				diffy = abs(bbi.y - bbj.y)
				if ciou == 0 and (diffx + diffy > 10) or cgap > 25:
					# print(seqi,seqj,'ciou 0, dif',diffx,diffy)
					continue
				cbbfit = bbi.get_bb_fit(bbj)

				# closs = (1-ciou)*weightiou + cbbfit*weightbbfit + cgap*weightgap
				closs = (1 - ciou) * weightiou + cbbfit * weightbbfit + cgap * weightgap
				if DEBUG:
					print(seqi, '(%d,%d-%d,%d)' % (
					seqbb[seqi][-1][0] - seqbb[seqi][0][0], seqbb[seqi][0][0], seqbb[seqi][-1][0],
					self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].x),
						  seqj, '(%d,%d-%d,%d)' % (
						  seqbb[seqj][-1][0] - seqbb[seqj][0][0], seqbb[seqj][0][0], seqbb[seqj][-1][0],
						  self.bbr[seqbb[seqj][0][0]][seqbb[seqj][0][1]].x),
						  ciou, cbbfit, cgap * weightgap, closs)
				if closs < lossthres:
					if closs < cminloss:
						cminloss = closs
						seqminind = seqj

			if seqminind == -1:
				# print('all start of seq not fit')
				continue
			else:
				# print('End of ',seqi,' to start of ',seqminind,' has min loss, test other ends')
				slicetuplej = seqbb[seqminind][0]
				bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
				ciminloss = np.inf
				seqiminind = -1
				availseqii = [i for i in range(len(endseqbb)) if
							  endseqbb[i] <= startseqbb[seqminind] and i not in connectend + [seqminind]]
				for seqii in availseqii:
					# find seqii end bb
					slicetupleii = seqbb[seqii][-1]
					bbii = copy.deepcopy(self.bbr[slicetupleii[0]][slicetupleii[1]])
					# calculate loss
					ciou = bbii.get_expand_iou(bbj, self.iouexpandratio)
					cgap = startseqbb[seqminind] - endseqbb[seqii]
					diffx = abs(bbii.x - bbj.x)
					diffy = abs(bbii.y - bbj.y)
					if ciou == 0 and (diffx > 10 or diffy > 10) or cgap > 25:
						continue
					cbbfit = bbii.get_bb_fit(bbj)

					closs = (1 - ciou) * weightiou + cbbfit * weightbbfit + cgap * weightgap
					if DEBUG:
						print('R', seqii, seqminind, ciou, cbbfit, cgap * weightgap, closs)
					if closs < lossthres:
						if closs < ciminloss:
							ciminloss = closs
							seqiminind = seqii
				if seqiminind == seqi:
					if DEBUG:
						print('back match, connect', seqi, seqminind)
					# connect from i to seqminind
					connectend.append(seqi)
					connectstart.append(seqminind)
				else:
					if DEBUG:
						print('back mismatch', seqi, seqminind, seqminind)

		# deep copy
		bbint = copy.deepcopy(self.bbr)
		seqbbint = []
		# make connection with start not in connect start
		for seqi in range(len(seqbb)):
			if seqi in connectstart:
				continue
			seqbbint.append(copy.deepcopy(seqbb[seqi]))
			cseq = seqi
			while cseq in connectend:
				nextseq = connectstart[connectend.index(cseq)]
				if DEBUG:
					print('connect', cseq, nextseq)
				bbend = bbint[seqbb[cseq][-1][0]][seqbb[cseq][-1][1]].getbbclassflat()
				bbstart = self.bbr[seqbb[nextseq][0][0]][seqbb[nextseq][0][1]].getbbclassflat()
				if DEBUG:
					print('bbstart', bbstart, 'end', bbend)
				bbdiff = [bbstart[i] - bbend[i] for i in range(len(bbend))]
				gap = startseqbb[nextseq] - endseqbb[seqi]
				if DEBUG:
					print('gap', gap, 'diff', bbdiff)
				for coni in range(endseqbb[seqi] + 1, startseqbb[nextseq]):
					if DEBUG:
						print('appending slice', coni)
						print([bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))])
					seqbbint[-1].append((coni, len(bbint[coni])))
					intpbblist = [bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))]
					bbint[coni].append(BB.fromlistclass(intpbblist))
					if DEBUG:
						print(len(seqbb[0]), 'cseq')
				seqbbint[-1].extend(seqbb[nextseq])
				# update end loc
				endseqbb[seqi] = endseqbb[nextseq]
				cseq = nextseq

		if self.RESTRICT_BRANCH_NUM == 0:
			self.seqbbsim = copy.deepcopy(seqbbint)
		else:
			# remove short seq
			seqbbsim = []
			thresremovebranch = 20
			seqmax = np.array([len(seqbbint[i]) for i in range(len(seqbbint))]).argsort()[-pn:][::-1]
			xpos0 = [bbint[seqbbint[seqmax[0]][seqslicei][0]][seqbbint[seqmax[0]][seqslicei][1]].x for seqslicei in
					 range(len(seqbbint[seqmax[0]]))]
			meanx0 = np.mean(xpos0)
			stdx0 = np.std(xpos0)
			xpos1 = [bbint[seqbbint[seqmax[1]][seqslicei][0]][seqbbint[seqmax[1]][seqslicei][1]].x for seqslicei in
					 range(len(seqbbint[seqmax[1]]))]
			meanx1 = np.mean(xpos1)
			stdx1 = np.std(xpos1)
			print('Top tracklets', 'len', len(seqbbint[seqmax[0]]), 'xmean', meanx0, '+-', stdx0, 'len',
				  len(seqbbint[seqmax[1]]), 'xmean', meanx1, '+-', stdx1)
			for seqi in range(len(seqbbint)):
				if len(seqbbint[seqi]) < 3:
					if DEBUG:
						print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
						seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
						self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
							  'with', len(seqbbint[seqi]), 'points')
					continue
				elif len(seqbbint[seqi]) < 5:
					sumartp = 0
					for si in range(len(seqbbint[seqi])):
						sumartp += bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]].get_score()
					if sumartp < 0.5 * len(seqbbint[seqi]):
						if DEBUG:
							print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
							seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
							self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
								  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
						continue
					else:
						if DEBUG:
							print('keep seqi', seqi, '(%d,%d-%d,%d)' % (
							seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
							self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
								  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
				# for longer seqs, stricter rule if not belong to prior defined number of tracklets pn
				elif pn > 0 and seqi not in np.array([len(seqbbint[i]) for i in range(len(seqbbint))]).argsort()[-pn:][
											::-1]:
					sumartp = 0
					for si in range(len(seqbbint[seqi])):
						# if seqbbint[seqi][si][1]<len(bbr[seqbbint[seqi][si][0]]) and \
						#	bbr[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]]==bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]]:
						sumartp += bbint[seqbbint[seqi][si][0]][seqbbint[seqi][si][1]].get_score()
					if sumartp < 0.35 * len(seqbbint[seqi]):
						print('skip seqi', seqi, '(%d,%d-%d,%d)' % (
						seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
						self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
							  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))
						continue
					else:
						print('keep seqi', seqi, '(%d,%d-%d,%d)' % (
						seqbbint[seqi][-1][0] - seqbbint[seqi][0][0], seqbbint[seqi][0][0], seqbbint[seqi][-1][0],
						self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
							  'with', len(seqbbint[seqi]), 'points', 'ART poss avg', sumartp / len(seqbbint[seqi]))

				if self.usebranch and pn > 0:

					if seqi not in seqmax:
						xcenter = []
						for seqslicei in range(len(seqbbint[seqi])):
							box = bbint[seqbbint[seqi][seqslicei][0]][seqbbint[seqi][seqslicei][1]]
							xcenter.append(box.x)
						meancenter = np.mean(xcenter)
						mindis0 = abs(meancenter - meanx0)
						mindis1 = abs(meancenter - meanx1)
						# if np.mean(xcenter)>330 and np.mean(xcenter)<390:
						# 	if DEBUG:
						# 		print(seqi,'(%d,%d-%d,%d)'%(seqbbint[seqi][-1][0]-seqbbint[seqi][0][0],seqbbint[seqi][0][0],seqbbint[seqi][-1][0],self.bbr[seqbbint[seqi][0][0]][seqbbint[seqi][0][1]].x),
						# 		  'center box,skip')
						# 	continue
						print(seqi, 'min dist to main branches', mindis0, mindis1)
						if mindis0 < 2 * stdx0 or mindis1 < 2 * stdx1:
							print('Keep')
						else:
							continue
						USE3DDIST = 0
						# 3d distance to two main branches
						if USE3DDIST:
							cbranchbb0s = [seqbbint[seqi][0][0]] + bbint[seqbbint[seqi][0][0]][
																	   seqbbint[seqi][0][1]].getbb()[:2]
							cbranchbb0e = [seqbbint[seqi][-1][0]] + bbint[seqbbint[seqi][-1][0]][
																		seqbbint[seqi][-1][1]].getbb()[:2]

							mindis = 99999
							for slicei in range(len(seqbbint[seqmax[0]])):
								maxbranchbb = [seqbbint[seqmax[0]][slicei][0]] + bbint[seqbbint[seqmax[0]][slicei][0]][
																					 seqbbint[seqmax[0]][slicei][
																						 1]].getbb()[:2]
								cdists = np.sqrt(
									np.sum([(cbranchbb0s[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
								cdiste = np.sqrt(
									np.sum([(cbranchbb0e[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
								if cdists < mindis:
									mindis = cdists
								if cdiste < mindis:
									mindis = cdiste
							for slicei in range(len(seqbbint[seqmax[1]])):
								maxbranchbb = [seqbbint[seqmax[1]][slicei][0]] + bbint[seqbbint[seqmax[1]][slicei][0]][
																					 seqbbint[seqmax[1]][slicei][
																						 1]].getbb()[:2]
								cdists = np.sqrt(
									np.sum([(cbranchbb0s[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
								cdiste = np.sqrt(
									np.sum([(cbranchbb0e[i] - maxbranchbb[i]) ** 2 for i in range(len(maxbranchbb))]))
								if cdists < mindis:
									mindis = cdists
								if cdiste < mindis:
									mindis = cdiste
							print(seqi, 'min dist to main branches', mindis)
							if mindis > thresremovebranch:
								continue
							else:
								print('Keep')
				self.seqbbsim.append(copy.deepcopy(seqbbint[seqi]))

		if self.RESTRICT_BRANCH_NUM == 1 and pn == 1:
			# further refine per slice bb number
			self.seqbbsimpn = []
			artsumartp = []
			for seqi in range(len(self.seqbbsim)):
				sumartp = 0
				for si in range(len(self.seqbbsim[seqi])):
					sumartp += bbint[self.seqbbsim[seqi][si][0]][self.seqbbsim[seqi][si][1]].get_score()
				artsumartp.append(sumartp)
			ordartsum = np.argsort(artsumartp)[::-1]
			bbnumseq = np.zeros((len(bbint)))
			bbnumseqid = np.zeros((len(bbint)), dtype=np.uint8)
			for seqi in range(len(self.seqbbsim)):
				if seqi in ordartsum[:pn]:
					for se in self.seqbbsim[seqi]:
						bbnumseq[se[0]] += 1
						bbnumseqid[se[0]] = se[1]
			if DEBUG:
				print(bbnumseq, bbnumseqid)
			for seqi in range(len(self.seqbbsim)):
				if seqi not in ordartsum[:pn]:
					# copy with missing bb
					bbadd = []
					for se in self.seqbbsim[seqi]:
						if bbnumseq[se[0]] < pn:
							# if DEBUG:
							#	print(se,'extra bb added')
							nearestbb = None
							for ofi in range(1, len(bbint)):
								if se[0] - ofi >= 0 and bbnumseq[se[0] - ofi] > 0:
									nearestbb = bbint[int(se[0] - ofi)][bbnumseqid[int(se[0] - ofi)]]
									break
								if se[0] + ofi < len(bbnumseq) and bbnumseq[se[0] + ofi] > 0:
									nearestbb = bbint[int(se[0] + ofi)][bbnumseqid[int(se[0] + ofi)]]
									break
							if nearestbb is not None:
								print(bbint[se[0]][se[1]], 'nearestbb', nearestbb)
								fitbbnei = bbint[se[0]][se[1]].get_bb_fit(nearestbb)
								print(se, 'fitbbnei', fitbbnei)
								if fitbbnei < 20:
									bbadd.append(se)
									bbnumseq[se[0]] += 1
									bbnumseqid[se[0]] = se[1]
							else:
								print('No neigh...not possible', se[0])
					if len(bbadd):
						self.seqbbsimpn.append(bbadd)
						if DEBUG:
							print('add extra bb, len', len(bbadd))
				else:
					self.seqbbsimpn.append(copy.deepcopy(self.seqbbsim[seqi]))
			self.seqbbsim = self.seqbbsimpn

		self.bbint = bbint
		# update bbs with only valid seqsim
		self.bbs = [[] for i in range(len(bbint))]
		for seqi in range(len(self.seqbbsim)):
			for i in range(len(self.seqbbsim[seqi])):
				slicei = self.seqbbsim[seqi][i][0]
				orislicebbid = self.seqbbsim[seqi][i][1]
				self.seqbbsim[seqi][i] = (slicei, len(self.bbs[slicei]))
				self.bbs[slicei].append(copy.deepcopy(bbint[slicei][orislicebbid]))

		if self.trackletreffigpath is not None:
			self.drawrfseqbb(savename=self.trackletreffigpath + 'ref.png')
		else:
			self.drawrfseqbb()

		# adjust boundary of bb to edge
		if boundaryadjustment:
			self.ba()
			self.interpseqneighgap()
		# remove bb with similar detection
		self.removeduplicatebb()

		self.smoothbb(fig=0)
		if self.trackletreffigpath is not None:
			self.drawrfseqbb(savename=self.trackletreffigpath + 'refba.png')
		else:
			self.drawrfseqbb()

	def smoothbb(self, fig=1):
		seqbbsim = self.seqbbsim
		dim = ['x', 'y', 'w', 'h']
		col = len(dim) + self.classsize
		# smooth trace
		if fig:
			pyplot.figure(figsize=(15, 10))
		for fi in range(len(dim) + self.classsize):
			for seqi in range(len(seqbbsim)):
				if len(seqbbsim[seqi]) < self.medks:
					# if fi==0:
					#	print(seqi,'len<',self.medks,', no smooth')
					continue
				# get all slice id as y axis
				sliceq = [seqbbsim[seqi][i][0] for i in range(len(seqbbsim[seqi]))]
				if fi < 4:
					yseq = [getattr(self.bbs[seqbbsim[seqi][slicei][0]][seqbbsim[seqi][slicei][1]], dim[fi]) for slicei
							in range(len(sliceq))]
				else:
					yseq = [self.bbs[seqbbsim[seqi][slicei][0]][seqbbsim[seqi][slicei][1]].classes[fi - 4] for slicei in
							range(len(sliceq))]
				yseqsmooth = scipy.signal.medfilt(yseq, kernel_size=self.medks)

				# smooth using mean of neighbor
				yseqsmoothmean = []
				for si in range(len(yseqsmooth)):
					meany = [yseqsmooth[si]]
					cslicei = seqbbsim[seqi][si][0]
					for addi in range(1, self.smoothfilterhsz + 1):
						if si - addi > 0:
							slicei = seqbbsim[seqi][si - addi][0]
							if cslicei - slicei == addi:
								meany.append(yseqsmooth[si - addi])
						if si + addi < len(yseqsmooth):
							slicei = seqbbsim[seqi][si + addi][0]
							if slicei - cslicei == addi:
								meany.append(yseqsmooth[si + addi])
					yseqsmoothmean.append(np.mean(meany))
				yseqsmooth = yseqsmoothmean
				# yseqsmooth = [yseqsmooth[0]]+[np.mean(yseqsmooth[i-1:i+2]) for i in range(1,len(yseqsmooth)-1)]+[yseqsmooth[-1]]
				if fig:
					ax1 = pyplot.subplot(2, col, fi + 1)
					ax1.plot(yseq, sliceq, 'o', label='Trace%d' % seqi)
					# ax1.set_ylim(483.3, 344.7)
					ax2 = pyplot.subplot(2, col, col + fi + 1)
					ax2.plot(yseqsmooth, sliceq, 'o', label='Trace%d' % seqi)
				# record in bbs
				for i in range(len(sliceq)):
					slicei = sliceq[i]
					slicebbid = seqbbsim[seqi][i][1]
					if fi < 4:
						setattr(self.bbs[slicei][slicebbid], dim[fi], yseqsmooth[i])
					else:
						self.bbs[slicei][slicebbid].classes[fi - 4] = yseqsmooth[i]

			if fig:
				# pyplot.legend(loc="lower right")
				if fi < 4:
					ax1.set_title(dim[fi] + " Before")
					ax2.set_title(dim[fi] + " After")
				else:
					ax1.set_title('class %d Before' % (fi - 4))
					ax2.set_title('class %d After' % (fi - 4))
				# pyplot.xlim([200,600])
				ax1.set_ylim(max(ax1.get_ylim()), min(ax1.get_ylim()))
				ax2.set_ylim(max(ax1.get_ylim()), min(ax1.get_ylim()))
		# pyplot.gca().invert_yaxis()

		if fig:
			pyplot.show()

	def setresolution(self, res):
		self.RESOLUTION = res

	def settrackletreffigpath(self, trackletreffigpath):
		self.trackletreffigpath = trackletreffigpath

	def ba(self):
		# boundary adjustment
		DEBUG = 0
		if self.imgstack is None:
			print('No image stack, skip boundary adjustment')
			return
		if DEBUG:
			# x y positions for 3 situations(before, after baref, and after smooth)
			xypos = np.zeros((len(self.bbs), 3, 2))
		NEEDSEG = 0
		if self.gwsegstack is None:
			NEEDSEG = 1
			self.gwsegstack = np.zeros((self.imgstack.shape))
		else:
			print('use seg for ba')
		# use great wall seg for boundary info
		gwmodel = self.gwmodel
		assert self.gwmodel is not None
		seqbbsim = self.seqbbsim
		seqbbsimori = copy.deepcopy(seqbbsim)

		for seqi in range(len(seqbbsim)):
			for seqslicei in range(len(seqbbsim[seqi]) - 1, -1, -1):
				if seqslicei % 5 == 0:
					print('\rBA', seqi, '/', len(seqbbsim), seqslicei, '/', len(seqbbsim[seqi]), end="")
				slicei = seqbbsim[seqi][seqslicei][0]
				bbi = seqbbsim[seqi][seqslicei][1]
				cbb = self.bbs[slicei][bbi]
				rx = int(round(cbb.x))
				ry = int(round(cbb.y))
				if self.RESOLUTION == -1:
					cpsz = 32
				else:
					cpsz = int(np.ceil(32 / (self.RESOLUTION / gwmodel.gwmodelRes)))
				if NEEDSEG == 0:
					probpatchnorm = croppatch(self.gwsegstack[slicei], cbb.y, cbb.x, cpsz, cpsz)
					probpatch = probpatchnorm / 255
				else:
					dcmpatch = croppatch(self.imgstack[slicei], cbb.y, cbb.x, cpsz, cpsz)
					if np.max(dcmpatch) != 0:
						dcmpatchnorm = dcmpatch / np.max(dcmpatch)
					else:
						print('max pixel zeros at cropped patch')
						continue

					# use great wall segmentation to generate a rough prob map for center adjustment
					if self.RESOLUTION == -1:
						dcmres = -1
					else:
						dcmres = self.RESOLUTION
					probpatch = gwmodel.predprob(dcmpatchnorm, dcmres)
					probpatchnorm = probpatch / np.max(probpatch) * 255
					# fill vessel image stack
					self.gwsegstack[slicei] = fillpatch(self.gwsegstack[slicei], probpatchnorm, ry, rx)
				# self.gwsegstack[slicei,ry-halfpatchsize:ry+halfpatchsize,rx-halfpatchsize:rx+halfpatchsize] = probpatchnorm
				halfpatchsize = probpatchnorm.shape[0] // 2
				if DEBUG:
					# paint before and after adjustment
					xypos[slicei, 0] = self.bbs[slicei][bbi].getbb()[0:2]
					pyplot.figure(figsize=(10, 10))
					cbblist = [int(xypos[slicei][0][i]) for i in range(2)]
					probimgpatch = croppatch(self.imgstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
											 halfpatchsize)
					probimgpatch[halfpatchsize, halfpatchsize - 1:halfpatchsize + 2] = 255
					probimgpatch[halfpatchsize - 1:halfpatchsize + 2, halfpatchsize] = 255
					pyplot.subplot(2, 2, 1)
					pyplot.title('prob ori')
					pyplot.imshow(probimgpatch)
					segimgpatch = croppatch(self.gwsegstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
											halfpatchsize)
					pyplot.subplot(2, 2, 2)
					pyplot.title('prob ori seg')
					pyplot.imshow(segimgpatch)

				# predict boundary adjustment
				adjlist = self._refct2(probpatch)
				if adjlist is None:
					# print('remove',seqi,seqslicei,slicei,rx)
					# pyplot.show()
					del seqbbsim[seqi][seqslicei]
					continue
				if DEBUG:
					print(slicei, adjlist)
				self.bbs[slicei][bbi].x = rx + adjlist[0]
				self.bbs[slicei][bbi].y = ry + adjlist[1]
				if DEBUG:
					xypos[slicei, 1] = self.bbs[slicei][bbi].getbb()[0:2]
					cbblist = [int(xypos[slicei][1][i]) for i in range(2)]
					probimgpatch = croppatch(self.imgstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
											 halfpatchsize)
					probimgpatch[halfpatchsize, halfpatchsize - 1:halfpatchsize + 2] = 255
					probimgpatch[halfpatchsize - 1:halfpatchsize + 2, halfpatchsize] = 255
					pyplot.subplot(2, 2, 3)
					pyplot.title('prob refct %d %d' % (
					xypos[slicei][1][0] - xypos[slicei][0][0], xypos[slicei][1][1] - xypos[slicei][0][1]))
					pyplot.imshow(probimgpatch)
					segimgpatch = croppatch(self.gwsegstack[slicei], cbblist[1], cbblist[0], halfpatchsize,
											halfpatchsize)
					pyplot.subplot(2, 2, 4)
					pyplot.title('prob refct seg')
					pyplot.imshow(segimgpatch)
					pyplot.show()

			if len(seqbbsim[seqi]) > 100:
				print('seq', seqi, 'main branch, no remove')
				seqbbsim[seqi] = seqbbsimori[seqi]

		# smooth again after center adjustment
		if DEBUG:
			self.smoothbb(fig=1)
		else:
			self.smoothbb(fig=0)

	# assume only one bounding box per slice
	# if DEBUG:
	# 	for slicei in range(len(self.bbs)):
	# 		xypos[slicei,2] = self.bbs[slicei][0].getbb()[0:2]
	# if DEBUG:
	# 	#display difference of pos before after ref ct
	# 	for slicei in range(len(self.bbs)):
	# 		pyplot.figure(figsize=(10,4))
	# 		cbblist = [int(xypos[slicei][0][i]) for i in range(2)]
	# 		print(slicei,cbblist)
	# 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
	# 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
	# 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
	# 		pyplot.subplot(1,3,1)
	# 		pyplot.title('prob ori')
	# 		pyplot.imshow(probimgpatch)

	# 		cbblist = [int(xypos[slicei][1][i]) for i in range(2)]
	# 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
	# 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
	# 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
	# 		pyplot.subplot(1,3,2)
	# 		pyplot.title('prob refct %d %d'%(xypos[slicei][1][0]-xypos[slicei][0][0],xypos[slicei][1][1]-xypos[slicei][0][1]))
	# 		pyplot.imshow(probimgpatch)

	# 		cbblist = [int(xypos[slicei][2][i]) for i in range(2)]
	# 		probimgpatch = np.copy(self.gwsegstack[slicei,cbblist[1]-halfpatchsize:cbblist[1]+halfpatchsize,cbblist[0]-halfpatchsize:cbblist[0]+halfpatchsize])
	# 		probimgpatch[halfpatchsize,halfpatchsize-1:halfpatchsize+2]=255
	# 		probimgpatch[halfpatchsize-1:halfpatchsize+2,halfpatchsize]=255
	# 		pyplot.subplot(1,3,3)
	# 		pyplot.title('prob smooth %d %d'%(xypos[slicei][2][0]-xypos[slicei][0][0],xypos[slicei][2][1]-xypos[slicei][0][1]))
	# 		pyplot.imshow(probimgpatch)
	# 		pyplot.show()

	def _refct2(self, probimg):
		maxtl = 0
		gap = 3
		mcy = probimg.shape[0] // 2
		mcx = probimg.shape[1] // 2

		nd = scipy.stats.norm(13, 5)
		pdf = [nd.pdf(i) for i in range(int(probimg.shape[0] * 1.5))]
		for i in range(13):
			pdf[i] = pdf[13]

		for yi in range(2, probimg.shape[0] - 2, gap):
			for xi in range(2, probimg.shape[1] - 2, gap):
				cx1 = probimg[yi::-1, xi]
				cx2 = probimg[yi:, xi]
				cy1 = probimg[yi, xi::-1]
				cy2 = probimg[yi, xi:]
				# RB
				cxy1 = [probimg[yi + i, xi + i] for i in range(min(probimg.shape[0] - yi, probimg.shape[1] - xi))]
				# RU
				cxy2 = [probimg[yi - i, xi + i] for i in range(min(yi, probimg.shape[1] - xi))]
				# LB
				cxy3 = [probimg[yi + i, xi - i] for i in range(min(probimg.shape[0] - yi, xi))]
				# LU
				cxy4 = [probimg[yi - i, xi - i] for i in range(min(yi, xi))]
				ctint = probimg[yi, xi]
				lcx1 = np.max([i * abs(cx1[i] - cx1[i - 1]) * pdf[i] for i in range(1, len(cx1))])
				lcx2 = np.max([i * abs(cx2[i] - cx2[i - 1]) * pdf[i] for i in range(1, len(cx2))])
				lcy1 = np.max([i * abs(cy1[i] - cy1[i - 1]) * pdf[i] for i in range(1, len(cy1))])
				lcy2 = np.max([i * abs(cy2[i] - cy2[i - 1]) * pdf[i] for i in range(1, len(cy2))])
				lcxy1 = np.max([abs(cxy1[i] - cxy1[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy1))])
				lcxy2 = np.max([abs(cxy2[i] - cxy2[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy2))])
				lcxy3 = np.max([abs(cxy3[i] - cxy3[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy3))])
				lcxy4 = np.max([abs(cxy4[i] - cxy4[i - 1]) * pdf[int(i * 1.4)] for i in range(1, len(cxy4))])
				tl = lcx1 * lcx2 * lcy1 * lcy2 * lcxy1 * lcxy2 * lcxy3 * lcxy4 * (1 - ctint)
				if tl > maxtl:
					# print(xi,yi,tl)
					maxtl = tl
					mcx = xi
					mcy = yi
		# print('maxtl',maxtl)
		if maxtl < 1e-10:
			# print('maxtl',maxtl)
			return None
		else:
			return [mcx - probimg.shape[1] // 2, mcy - probimg.shape[0] // 2]

	def removeduplicatebb(self):
		# remove detection overlaps
		# start from longest seq, remove bb from other seq if overlap
		argseqlen = np.argsort([len(seqi) for seqi in self.seqbbsim])[::-1]
		for seqidx in range(len(argseqlen)):
			seqid = argseqlen[seqidx]
			seqsegconfs = []
			for seqbbid in range(len(self.seqbbsim[seqid]) - 1, -1, -1):
				slicei = self.seqbbsim[seqid][seqbbid][0]
				bbid = self.seqbbsim[seqid][seqbbid][1]
				bbi = self.bbs[slicei][bbid]
				# search other shorter seq
				for seqjdx in range(seqidx + 1, len(argseqlen)):
					seqjd = argseqlen[seqjdx]
					for seqbbjd in range(len(self.seqbbsim[seqjd]) - 1, -1, -1):
						if slicei == self.seqbbsim[seqjd][seqbbjd][0]:
							bbjd = self.seqbbsim[seqjd][seqbbjd][1]
							bbj = self.bbs[slicei][bbjd]
							bbiom = bbi.get_iom(bbj)
							if bbiom > 0.5:
								del self.seqbbsim[seqjd][seqbbjd]
						# print('remove duplicate bb',slicei,seqid,seqbbid,seqjd,seqbbjd)
		# update bbs after removing some items in seqbb
		self.refreshseq()

	def refreshseq(self):
		if self.bbs is None:
			print('no bbs')
			return
		bbsnew = [[] for i in range(len(self.bbs))]
		seqbbsim = copy.deepcopy(self.seqbbsim)
		for seqid in range(len(seqbbsim)):
			for seqbbid in range(len(seqbbsim[seqid])):
				slicei = seqbbsim[seqid][seqbbid][0]
				bbid = seqbbsim[seqid][seqbbid][1]
				seqbbsim[seqid][seqbbid] = (slicei, len(bbsnew[slicei]))
				bbsnew[slicei].append(copy.deepcopy(self.bbs[slicei][bbid]))
		self.bbs = bbsnew
		self.seqbbsim = seqbbsim

	def interpseqneighgap(self):
		# interpolate the gap casued by ba remove bb within each seqbbsim
		# if there are neighboring bbs within 5 slices, use mean of all neigbors as the result
		newseqbb = [[] for i in range(len(self.seqbbsim))]
		for seqid in range(len(self.seqbbsim)):
			if len(self.seqbbsim[seqid]) == 0:
				continue
			startslicei = self.seqbbsim[seqid][0][0]
			endslicei = self.seqbbsim[seqid][-1][0]
			for slicei in range(startslicei, endslicei + 1):
				# search existing seqbbsim
				fd = -1
				bbneiitems = []
				for seqbbid in range(len(self.seqbbsim[seqid])):
					slicej = self.seqbbsim[seqid][seqbbid][0]
					if slicej == slicei:
						newseqbb[seqid].append(self.seqbbsim[seqid][seqbbid])
						fd = slicej
						break
					if abs(slicei - slicej) < 5:
						sliceni = self.seqbbsim[seqid][seqbbid][0]
						bbnei = self.seqbbsim[seqid][seqbbid][1]
						bbitem = self.bbs[sliceni][bbnei].getbbclassflat()
						bbneiitems.append(bbitem)

				if fd != -1:
					continue
				# if not found, inter if there are neighbor bb
				elif fd == -1 and len(bbneiitems):
					meanbbitem = []
					for fi in range(len(bbneiitems[0])):
						meanbbitem.append(np.mean([bbneiitems[bbni][fi] for bbni in range(len(bbneiitems))]))
					newseqbb[seqid].append((slicei, len(self.bbs[slicei])))
					self.bbs[slicei].append(BB.fromlistclass(meanbbitem))
		# else:
		# print(seqid,slicei,'gap interp')
		self.seqbbsim = newseqbb

	def getbbs(self, clabel=0, minmax=0):
		if self.bbs is None:
			print('Refine first')
			return
		bbs = [[] for i in range(len(self.bbs))]
		for slicei in range(len(self.bbs)):
			bbsslice = []
			for bbi in range(len(self.bbs[slicei])):
				if clabel == 0:
					if minmax == 0:
						bbsslice.append(self.bbs[slicei][bbi].getbb())
					else:
						bbsslice.append(self.bbs[slicei][bbi].getminmax())
				else:
					if minmax == 0:
						bbsslice.append(self.bbs[slicei][bbi].getbbclabel())
					else:
						bbsslice.append(self.bbs[slicei][bbi].getminmaxclabel())
			if len(bbsslice):
				bbs[slicei] = bbsslice
		return bbs

	def getbbsclass(self):
		if self.bbs is None:
			print('Refine first')
			return
		bbs = [[] for i in range(len(self.bbs))]
		for slicei in range(len(self.bbs)):
			bbsslice = []
			for bbi in range(len(self.bbs[slicei])):
				bbsslice.extend(self.bbs[slicei][bbi].getbbclass())
			if len(bbsslice):
				if bbsslice[0] == 0:
					print('bbsslice 0')
					continue
				bbs[slicei].append(bbsslice)
		return bbs

	# def fillmissing(self):
	# 	if self.bbs is None:
	# 		print('Refine first')
	# 		return
	# 	bbs=[[] for i in range(len(self.bbs))]
	# 	for slicei in range(len(self.bbs)):

	# write refined boxes to bbrfdir
	def writebbs(self, bbrffolder, einame):
		if not os.path.exists(bbrffolder):
			os.mkdir(bbrffolder)
		else:
			for fi in os.listdir(bbrffolder):
				os.remove(bbrffolder + '/' + fi)
			print('remove existing bbs')
		bbs = self.bbs
		for slicei in range(len(bbs)):
			slicename = os.path.join(bbrffolder, einame + '_%03d' % slicei + '.txt')
			with open(slicename, 'w') as file:
				file.write('%d\n' % len(bbs[slicei]))
				for bbiBB in bbs[slicei]:
					bbi = bbiBB.getminmaxclabel()
					# print(slicei,box)
					# label tool format
					file.write('%.2f %.2f %.2f %.2f %.2f %d\n' % (bbi[0], bbi[1], bbi[2], bbi[3], bbi[4], bbi[5]))

	def writebbr(self, bbfolder, einame):
		if not os.path.exists(bbfolder):
			os.mkdir(bbfolder)
		else:
			print('remove existing bbr', os.listdir(bbfolder))
			for fi in os.listdir(bbfolder):
				os.remove(bbfolder + '/' + fi)

		bbr = self.bbr
		for slicei in range(len(bbr)):
			slicename = os.path.join(bbfolder, einame + '_%03d' % slicei + '.txt')
			with open(slicename, 'w') as file:
				file.write('%d\n' % len(bbr[slicei]))
				for bbiBB in bbr[slicei]:
					bbi = bbiBB.getminmaxclabel()
					# print(slicei,box)
					# label tool format
					file.write('%.2f %.2f %.2f %.2f %.2f %d\n' % (bbi[0], bbi[1], bbi[2], bbi[3], bbi[4], bbi[5]))

	# connect bb using minimum intensity path
	def connectbb(self, featuremodel=None, config=None, lossthres = 1):
		DEBUG = 1
		if config is None:
			pass
		else:
			if 'debug' in config:
				DEBUG = config['debug']
		if DEBUG:
			self.drawseqbb()
		self.seqbbsim = []
		seqbb = copy.deepcopy(self.seqbb)
		# starting/ending slice id of each seq
		startseqbb = [seqbb[i][0][0] for i in range(len(seqbb))]
		endseqbb = [seqbb[i][-1][0] for i in range(len(seqbb))]

		connectend = []
		connectstart = []
		# find seqbb after end of each seqbb
		for seqi in range(len(seqbb)):
			# skip if already end of slice
			if endseqbb[seqi] == len(self.bbr):
				continue

			# already has connection
			if seqi in connectend:
				continue
			if DEBUG:
				print('seqid', seqi, 'slicei', seqbb[seqi][0][0], '-', seqbb[seqi][-1][0], 'bb start',
					  self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].getbb())

			cminloss = np.inf
			seqminind = -1

			availseqj = [i for i in range(len(startseqbb)) if
						 startseqbb[i] >= endseqbb[seqi] and i not in connectstart + [seqi]]
			if DEBUG:
				print('avail seqj', availseqj)
			for seqj in availseqj:
				# find seqi end bb
				slicetuplei = seqbb[seqi][-1]
				bbi = copy.deepcopy(self.bbr[slicetuplei[0]][slicetuplei[1]])
				# find seqj start bb
				slicetuplej = seqbb[seqj][0]
				bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
				# calculate loss
				cgap = startseqbb[seqj] - endseqbb[seqi]
				diffx = abs(bbi.x - bbj.x)
				diffy = abs(bbi.y - bbj.y)
				if (diffx + diffy > 50) or cgap > 25:
					# print(seqi,seqj, 'dif',diffx,diffy)
					continue

				pos1 = [bbi.x, bbi.y, endseqbb[seqi]]
				pos2 = [bbj.x, bbj.y, startseqbb[seqj]]
				if featuremodel is None:
					intensity_along_path = np.array(intpath(pos1, pos2, self.imgstack))
					intensity_along_path_d = abs(intensity_along_path[1:] - intensity_along_path[:-1])
					closs = np.sum(intensity_along_path_d)
				else:
					if DEBUG:
						print(pos1, pos2)
					closs = bb_match_score(featuremodel, self.imgstack, pos1, pos2) / 20
				if DEBUG:
					print(seqi, '(%d,%d-%d,%d)' % (
						seqbb[seqi][-1][0] - seqbb[seqi][0][0], seqbb[seqi][0][0], seqbb[seqi][-1][0],
						self.bbr[seqbb[seqi][0][0]][seqbb[seqi][0][1]].x),
						  seqj, '(%d,%d-%d,%d)' % (
							  seqbb[seqj][-1][0] - seqbb[seqj][0][0], seqbb[seqj][0][0], seqbb[seqj][-1][0],
							  self.bbr[seqbb[seqj][0][0]][seqbb[seqj][0][1]].x),
						  cgap, closs,lossthres)
				if closs < lossthres:
					if closs < cminloss:
						cminloss = closs
						seqminind = seqj

			if seqminind == -1:
				# print('all start of seq not fit')
				continue
			else:
				# print('End of ',seqi,' to start of ',seqminind,' has min loss, test other ends')
				slicetuplej = seqbb[seqminind][0]
				bbj = copy.deepcopy(self.bbr[slicetuplej[0]][slicetuplej[1]])
				ciminloss = np.inf
				seqiminind = -1
				availseqii = [i for i in range(len(endseqbb)) if
							  endseqbb[i] <= startseqbb[seqminind] and i not in connectend + [seqminind]]
				if DEBUG:
					print('availseqii', availseqii)
				for seqii in availseqii:
					# find seqii end bb
					slicetupleii = seqbb[seqii][-1]
					bbii = copy.deepcopy(self.bbr[slicetupleii[0]][slicetupleii[1]])
					# calculate loss
					cgap = startseqbb[seqminind] - endseqbb[seqii]
					diffx = abs(bbii.x - bbj.x)
					diffy = abs(bbii.y - bbj.y)
					if diffx + diffy > 50 or cgap > 25:
						continue

					pos1 = [bbii.x, bbii.y, endseqbb[seqii]]
					pos2 = [bbj.x, bbj.y, startseqbb[seqminind]]
					if featuremodel is None:
						intensity_along_path = np.array(intpath(pos1, pos2, self.imgstack))
						intensity_along_path_d = abs(intensity_along_path[1:] - intensity_along_path[:-1])
						closs = np.sum(intensity_along_path_d)
					else:
						if DEBUG:
							print(pos1, pos2)
						closs = bb_match_score(featuremodel, self.imgstack, pos1, pos2) / 20
					if DEBUG:
						print('R', seqii, seqminind, cgap, closs)
					if closs < lossthres:
						if closs < ciminloss:
							ciminloss = closs
							seqiminind = seqii
				if seqiminind == seqi:
					if DEBUG:
						print('back match, connect', seqi, seqminind)
					# connect from i to seqminind
					connectend.append(seqi)
					connectstart.append(seqminind)
				else:
					if DEBUG:
						print('back mismatch', seqi, seqiminind, seqminind)

		# deep copy
		bbint = copy.deepcopy(self.bbr)
		seqbbint = []
		# make connection with start not in connect start
		for seqi in range(len(seqbb)):
			if seqi in connectstart:
				continue
			seqbbint.append(copy.deepcopy(seqbb[seqi]))
			cseq = seqi
			while cseq in connectend:
				nextseq = connectstart[connectend.index(cseq)]
				if DEBUG:
					print('connect', cseq, nextseq)
				bbend = bbint[seqbb[cseq][-1][0]][seqbb[cseq][-1][1]].getbbclassflat()
				bbstart = self.bbr[seqbb[nextseq][0][0]][seqbb[nextseq][0][1]].getbbclassflat()
				if DEBUG:
					print('bbstart', bbstart, 'end', bbend)
				bbdiff = [bbstart[i] - bbend[i] for i in range(len(bbend))]
				gap = startseqbb[nextseq] - endseqbb[seqi]
				if DEBUG:
					print('gap', gap, 'diff', bbdiff)
				for coni in range(endseqbb[seqi] + 1, startseqbb[nextseq]):
					if DEBUG:
						print('appending slice', coni)
						print([bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))])
					seqbbint[-1].append((coni, len(bbint[coni])))
					intpbblist = [bbend[i] + bbdiff[i] * (coni - endseqbb[seqi]) / gap for i in range(len(bbdiff))]
					bbint[coni].append(BB.fromlistclass(intpbblist))
					if DEBUG:
						print(len(seqbb[0]), 'cseq')
				seqbbint[-1].extend(seqbb[nextseq])
				# update end loc
				endseqbb[seqi] = endseqbb[nextseq]
				cseq = nextseq

		self.seqbbsim = copy.deepcopy(seqbbint)

		self.bbint = bbint
		# update bbs with only valid seqsim
		self.bbs = [[] for i in range(len(bbint))]
		for seqi in range(len(self.seqbbsim)):
			for i in range(len(self.seqbbsim[seqi])):
				slicei = self.seqbbsim[seqi][i][0]
				orislicebbid = self.seqbbsim[seqi][i][1]
				self.seqbbsim[seqi][i] = (slicei, len(self.bbs[slicei]))
				self.bbs[slicei].append(copy.deepcopy(bbint[slicei][orislicebbid]))

		# if self.trackletreffigpath is not None:
		#	self.drawrfseqbb(savename=self.trackletreffigpath + 'ref.png')
		if DEBUG:
			self.drawrfseqbb()

	def seqbbsim_stat(self):
		stats = []
		for seq in self.seqbbsim:
			xpos = []
			ypos = []
			cpos = []
			for seqi in seq:
				slicei = seqi[0]
				bbid = seqi[1]
				bb = self.bbs[slicei][bbid]
				xpos.append(bb.x)
				ypos.append(bb.y)
				cpos.append(bb.c)
			stats.append([np.mean(xpos), np.mean(ypos), np.mean(cpos), len(seq)])
		return np.array(stats)

	def label_carotid_art(self):
		seqstat = np.array(self.seqbbsim_stat())
		seqlen = seqstat[:, 2]
		arglen = np.argsort(seqlen)[::-1]

	def calseqproperty(self, neckxcenter=256):
		ARTTYPE = ['ICAL', 'ECAL', 'ICAR', 'ECAR']
		bbs = self.bbs
		seqbbsim = self.seqbbsim
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

			# if more than 2 on one side
			if len(Lobj) > 2:
				Lconfs = []
				for ai in range(len(Lobj)):
					Lconfs.append(confs[Lobj[ai]])
				arglconf = np.argsort(Lconfs)[::-1]
				Lobj = [Lobj[arglconf[0]]] + [Lobj[arglconf[1]]]
				print('Multiple bb on left side, top2', Lobj)
			if len(Robj) > 2:
				Rconfs = []
				for ai in range(len(Robj)):
					Rconfs.append(confs[Robj[ai]])
				print(Rconfs)
				argrconf = np.argsort(Rconfs)[::-1]
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
		# print(bblabel)

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

	def seqbbsim_iou_tracklet(self, seqid, bbcomp):
		seq = self.seqbbsim[seqid]
		bbiou = []
		for seqi in seq:
			slicei = seqi[0]
			bbid = seqi[1]
			bb = self.bbs[slicei][bbid]
			if len(bbcomp[slicei]) > 0:
				bbc = bbcomp[slicei][0]
			else:
				print('no bbc on slice', slicei)
				continue
			bbiou.append(bb.get_iou(bbc))
			print(slicei, bbiou[-1])
		return np.mean(bbiou)

	# get tracklet size of num from longest seqbbsim. if num on certain slice reaches, ignore other seqbbsim
	# c: 0-x 1-y 2-conf 3-length
	def collect_tracklet_by_c(self, num=1, c=2, thres = None, seq=False):
		Tbbstat = self.seqbbsim_stat()
		tracklet = [[] for i in range(len(self.bbs))]
		tbb_len_sort = np.argsort(Tbbstat[:, c])[::-1]
		print('mean track conf', Tbbstat[:, c], 'sort', tbb_len_sort)
		trackletseq = []
		for seqid in tbb_len_sort:
			if thres is not None and Tbbstat[seqid, c]<thres:
				continue
			cseq = []
			for seqi in self.seqbbsim[seqid]:
				slicei = seqi[0]
				if len(tracklet[slicei]) >= num:
					continue
				bbid = seqi[1]
				bb = self.bbs[slicei][bbid]
				cseq.append((slicei, len(tracklet[slicei])))
				tracklet[slicei].append(copy.copy(bb))
			trackletseq.append(cseq)
		if seq:
			return tracklet, trackletseq
		else:
			return tracklet
