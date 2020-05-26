import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import os

def DSC(labelimg,predict_img_thres):
    A = labelimg>0.5*np.max(labelimg)
    B = predict_img_thres>0.5*np.max(predict_img_thres)
    return 2*np.sum(A[A==B])/(np.sum(A)+np.sum(B))
    '''Narr=np.array(labelimg.reshape(-1)>0.5) != np.array(predict_img_thres.reshape(-1))
    FN=np.sum(np.logical_and(Narr, np.array(labelimg.reshape(-1)>0.9)))
    FP=np.sum(Narr)-FN
    Tarr=np.array(labelimg.reshape(-1)>0.5) == np.array(predict_img_thres.reshape(-1))
    TP=np.sum(np.logical_and(Tarr, np.array(labelimg.reshape(-1)>0.9)))
    TN=np.sum(Tarr)-TP
    return 2*TP/(2*TP+FP+FN)'''

def diffmap(A,B):
    diffmap = np.zeros((A.shape[0],A.shape[1],3))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j]==1 and B[i][j]==1:
                diffmap[i,j,2] = 1
            elif A[i][j]==1 and B[i][j]==0:
                diffmap[i,j,1] = 1
            elif A[i][j]==0 and B[i][j]==1:
                diffmap[i,j,0] = 1
    return diffmap


class Evalresult():
    def __init__(self, eval_result_filename=None):
        self.result = {}
        self.eval_result_filename = eval_result_filename
        self.load()

    def __repr__(self):
        return 'evaluated %d cases, %d slices, dsc:%.4f'%(len(self.result.keys()),self.ct,self.meandsc)

    @property
    def ct(self):
        if self._ct is None:
            self._ct = 0
            for ei in self.result:
                self._ct += len(self.result[ei])
        return self._ct

    @property
    def meandsc(self):
        dscs, _ = self.getdscs()
        return np.mean(dscs)

    def getdscs(self):
        dscs = []
        slicenames = []
        for ei in self.result:
            for slicei in self.result[ei]:
                if 'dsc' not in self.result[ei][slicei]:
                    continue
                dscs.append(self.result[ei][slicei]['dsc'])
                slicenames.append(slicei)
        return dscs, slicenames

    @property
    def meanet(self):
        ets = []
        for ei in self.result:
            for slicei in self.result[ei]:
                ets.append(self.result[ei][slicei]['et'])
        return np.mean(ets)

    #mean abs difference of predciton center
    @property
    def meanad(self):
        mads = []
        for ei in self.result:
            for slicei in self.result[ei]:
                mads.append(self.result[ei][slicei]['mad'])
        return np.mean(mads)

    @property
    def missnum(self):
        mnum = 0
        for ei in self.result:
            for slicei in self.result[ei]:
                if 'miss' in self.result[ei][slicei]:
                    mnum += 1
        return mnum

    def load(self):
        if os.path.exists(self.eval_result_filename):
            with open(self.eval_result_filename, 'rb') as fp:
                self.result = pickle.load(fp)
        self._ct = None

    def save(self):
        with open(self.eval_result_filename, 'wb') as fp:
            pickle.dump(self.result, fp)

    def eilist(self):
        return self.result.keys()

    def add(self, ei, slicename, obj):
        if ei not in self.result:
            self.result[ei] = {}
        self.result[ei][slicename] = obj
        pct = self.ct
        self._ct = pct + 1

    def append(self, ei, slicename, obj):
        if ei not in self.result:
            self.result[ei] = {}
        if slicename not in self.result[ei]:
            print('slicename not in result')
            return
        else:
            for item in obj:
                self.result[ei][slicename][item] = obj[item]


    def plt_hist(self):
        dsc, _ = self.getdscs()
        plt.hist(dsc, bins=20)
        plt.xlabel('Dice')
        plt.ylabel('Count')
        plt.show()

    def top_errs(self, n=20):
        dscs, slicenames = self.getdscs()
        n = min(n, len(dscs))
        argdsc = np.argsort(dscs)
        for ti in argdsc[:n]:
            cdsc = dscs[ti]
            slicename = slicenames[ti]
            print(cdsc, slicename)
            pjname, pename = slicename.split('-')
            einame = pjname + '-' + pename.split('I')[0] + slicename[-1]
            if 'polar_plot_name' in self.result[einame][slicename]:
                polar_plot = cv2.imread(self.result[einame][slicename]['polar_plot_name'])
                plt.figure(figsize=(20, 8))
                plt.axis('off')
                plt.imshow(polar_plot)
                plt.show()

from src.UTL import croppatch
from src.polarutil import plotct
class SegResult():
    def __init__(self, predname, examname, dicomstack):
        self.predname = predname
        self.examname = examname
        self.dicomstack = dicomstack

        segpath = self.predname + '/' + self.examname + '/' + self.examname + '-segresult.pickle'
        if os.path.exists(segpath):
            with open(segpath, 'rb') as fp:
                self.seg = pickle.load(fp)
            print('load from', segpath)
        else:
            print('init segresult')
            self.seg = {}
            self.seg['cartcont'] = {'lumen': [None for i in range(len(dicomstack))],
                                     'wall': [None for i in range(len(dicomstack))]}
            self.seg['polarcst'] = {'lumen': [None for i in range(len(dicomstack))],
                                     'wall': [None for i in range(len(dicomstack))]}
            self.seg['segct'] = [None for i in range(len(dicomstack))]

    def setdcmpath(self,dcmpath):
        self.dcmpath = dcmpath

    def addconts(self,slicei,contours,polarconsistency):
        if slicei<0 or slicei>=len(self.dicomstack):
            print('slicei invalid',slicei)
        self.seg['cartcont']['lumen'][slicei] = contours[0]
        self.seg['cartcont']['wall'][slicei] = contours[1]
        self.seg['polarcst']['lumen'][slicei] = polarconsistency[0]
        self.seg['polarcst']['wall'][slicei] = polarconsistency[1]

    def addcts(self,slicei,cts):
        self.seg['segct'][slicei] = cts

    def additem(self,slicei,item,value):
        if item not in self.seg:
            self.seg[item] = [None for i in range(len(self.dicomstack))]
        self.seg[item][slicei] = value

    def plotvw(self,slicei):
        if self.seg['cartcont']['lumen'][slicei] is None:
            return np.zeros((512,512))
        vw_seg = plotct(512, self.seg['cartcont']['lumen'][slicei], self.seg['cartcont']['wall'][slicei])
        return vw_seg

    def save(self):
        if not os.path.exists(self.predname + '/' + self.examname):
            os.mkdir(self.predname + '/' + self.examname)
        segpath = self.predname + '/' + self.examname + '/' + self.examname + '-segresult.pickle'
        with open(segpath, 'wb') as fp:
            pickle.dump(self.seg, fp)
        print('save to',segpath)

    def displayoff(self,figfilename=None):
        plt.figure(figsize=(18, 5))
        for slicei in range(len(self.seg['segct'])):
            if self.seg['segct'][slicei] is None:
                continue
            ct = np.mean(self.seg['segct'][slicei],axis=0)
            dcmpatch = croppatch(self.dicomstack[slicei], ct[1], ct[0], 64, 64)
            maxval = np.max(dcmpatch)
            dcmpatch[:,dcmpatch.shape[1]//2] = maxval
            dcmpatch[dcmpatch.shape[1] // 2] = maxval
            plt.subplot(2, int(np.ceil(len(self.dicomstack) / 2)), slicei + 1)
            plt.imshow(dcmpatch)
            plt.title(slicei)
        if figfilename is not None:
            plt.savefig(figfilename)
        else:
            plt.show()
        plt.close()

    def displayvw(self,figfilename=None):
        plt.figure(figsize=(18, 5))
        for slicei in range(len(self.seg['segct'])):
            if self.seg['segct'][slicei] is None or self.seg['cartcont']['lumen'][slicei] is None:
                continue
            ct = np.mean(self.seg['segct'][slicei],axis=0)
            dcmpatch = croppatch(self.plotvw(slicei), ct[1], ct[0], 64, 64)
            maxval = np.max(dcmpatch)
            dcmpatch[:,dcmpatch.shape[1]//2] = maxval
            dcmpatch[dcmpatch.shape[1] // 2] = maxval
            plt.subplot(2, int(np.ceil(len(self.dicomstack) / 2)), slicei + 1)
            plt.imshow(dcmpatch)
            plt.title(slicei)
        if figfilename is not None:
            plt.savefig(figfilename)
        else:
            plt.show()
        plt.close()

    def to_case_conts(self):
        #slice number, lumen/wall, cont point, x/y position
        caseconts = np.zeros((len(self.dicomstack),2,256,2))
        for slicei in range(len(self.seg['cartcont']['lumen'])):
            if self.seg['cartcont']['lumen'][slicei] is None:
                continue
            caseconts[slicei, 0] = self.seg['cartcont']['lumen'][slicei]
            caseconts[slicei, 1] = self.seg['cartcont']['wall'][slicei]
        return caseconts
