import numpy as np
import copy
from src.Tracklet import Tracklet
import matplotlib.pyplot as plt

class TrackletCT(Tracklet):
    def __init__(self, bbr, classsize=1, imgstack=None, iouthres=0.65):
        self.bbr = bbr
        self.gen_dist_tracklet(bbr)
        self.imgstack = imgstack
        assert imgstack.shape[0] == len(bbr)
        self.trackletreffigpath = None

    def gen_dist_tracklet(self,bbr):
        dstthres = 3
        # continuous sequence, each seq a list with each pt recorded as sliceid, bb id on that slice
        seqbb = []
        lbb = []  # last slice bounding box
        lbbid = []  # last slice bb id
        bifseqid = []
        for slicei in range(len(bbr)):
            # set available bb in this slice
            availbbslicei = [1 for i in range(len(bbr[slicei]))]
            # check last slice whether has continous seq
            lbbi = 0
            while lbbi < len(lbb):
                # find max iou for last slice bb
                cmindst = np.inf
                cmaxbbi = -1
                for bbi in range(len(bbr[slicei])):
                    cdst = bbr[slicei][bbi].dist(lbb[lbbi])
                    if cdst < cmindst:
                        cmindst = cdst
                        cmaxbbi = bbi
                if cmindst > dstthres:
                    # not found, stop seq
                    del lbb[lbbi]
                    del lbbid[lbbi]
                else:
                    # print(cmaxiou)
                    # update seqbb
                    seqbb[lbbid[lbbi]].append((slicei, cmaxbbi))
                    if availbbslicei[cmaxbbi] == 1:
                        availbbslicei[cmaxbbi] = 0
                        lbb[lbbi] = copy.deepcopy(bbr[slicei][cmaxbbi])
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
                    lbb.append(copy.deepcopy(bbr[slicei][avi]))
                    lbbid.append(len(seqbb))
                    seqbb.append([(slicei, avi)])
        print('seq length', len(seqbb))
        self.seqbb = seqbb
        return seqbb

    def drawseqbb(self):
        seqbb = self.seqbb
        bbct = self.bbr

        for seqid in range(len(seqbb)):
            xseq = []
            yseq = []
            sliceq = []
            seq = seqbb[seqid]
            for seqi in seq:
                slicei = seqi[0]
                bbid = seqi[1]
                bb = bbct[slicei][bbid]

                xseq.append(bb.x)
                yseq.append(bb.y)
                sliceq.append(slicei)

            plt.plot(xseq, sliceq, 'o', label='x-tracklet%d' % (seqid))
            #plt.plot(yseq, sliceq, 'o', label='y-seq%d' % (seqid))

            #print('xmean', np.mean(xseq))
            #print('ymean', np.mean(yseq))
        plt.legend()
        plt.ylabel('slice number', fontsize=18)
        plt.xlabel('x position', fontsize=18)
        plt.gca().invert_yaxis()
        plt.show()

