import glob
from src.variables import DATADESKTOPdir
import os
import numpy as np
import cv2
import pydicom
from src.UTL import croppatch

def adddb(dbloader,dbname,dbdir=None):
	if dbdir is None:
		dbdir = DATADESKTOPdir+'/DVWIMAGES/casepatch/Carotid'
	if not os.path.exists(dbdir+'/'+dbname):
		print('no dir exit',dbdir+'/'+dbname)
		return
	pilist = glob.glob(dbdir+'/'+dbname+'/list/*.list')
	for pi in pilist:
		pi = pi.replace('\\','/')
		pisplit = pi.split('/')
		casename = pisplit[-3]
		listname = pisplit[-1]
		piname = listname.split('E')[0]
		dbloader.addcase(casename+'-'+piname,pi)
	print('add',len(pilist),'list size',dbloader.size)

#read dcm image for testing
def load_dcm_stack(dicompath, norm=0, seq='101'):
	piname = dicompath.split('/')[-2]
	einame = dicompath.split('/')[-1]
	imgnamepattern = os.path.join(dicompath, einame + 'S' + seq + 'I*.dcm')
	targetimgs = glob.glob(imgnamepattern)
	if len(targetimgs) == 0:
		print('No dcm', imgnamepattern)
		return
	slices = [int(i.split('I')[-1][:-4]) for i in targetimgs]
	print(slices, np.min(slices), np.max(slices))
	cartimgstack = np.zeros((np.max(slices), 512, 512))
	imgnamepattern = os.path.join(dicompath, einame + 'S' + seq + 'I%d.dcm')
	if np.min(slices)!=1:
		print('min slice not 1')
	for slicei in range(np.min(slices), np.max(slices) + 1):
		imgfilename = imgnamepattern % slicei
		if os.path.exists(imgfilename):
			RefDs = pydicom.read_file(imgfilename).pixel_array
			if RefDs.shape != (512, 512):
				# print('Not regular size',RefDs.shape)
				if RefDs.shape[0] == RefDs.shape[1]:
					RefDs = cv2.resize(RefDs, (512, 512))
				else:
					padarr = np.zeros((max(RefDs.shape), max(RefDs.shape)))
					padarr[0:RefDs.shape[0], 0:RefDs.shape[1]] = RefDs
					RefDs = padarr
					RefDs = cv2.resize(RefDs, (512, 512))

			dcmimg = RefDs / np.max(RefDs)
			cartimgstack[slicei - 1] = dcmimg
		else:
			print('no slice img for', imgfilename)
	return cartimgstack

def crop_dcm_stack(dicomstack,sid,cty,ctx,hps,RESCALE,depth = 3):
	cartstack = croppatch(dicomstack[sid], cty, ctx, hps, hps)
	cartstackrz = cv2.resize(cartstack, (0,0), fx = RESCALE, fy = RESCALE)
	cartstackrz = np.repeat(cartstackrz[:, :, None], depth, axis=2)
	nei = depth // 2
	for ni in range(1, nei + 1):
		imgslicep = sid - ni
		if imgslicep >= 0:
			cartstack = croppatch(dicomstack[imgslicep], cty, ctx, 64, 64)
			cartstackrz[..., nei - ni] = cv2.resize(cartstack, (0,0), fx = RESCALE, fy = RESCALE)
		imgslicen = sid + ni
		if imgslicen < dicomstack.shape[0]:
			cartstack = croppatch(dicomstack[imgslicen], cty, ctx, 64, 64)
			cartstackrz[..., nei + ni] = cv2.resize(cartstack, (0,0), fx = RESCALE, fy = RESCALE)
	cartstackrz = cartstackrz / np.max(cartstackrz)
	return cartstackrz

import pickle
from src.UTL import croppatch, topolar
from src.polarutil import tocordpolar, toctbd, plotct
from src.mindist import find_nms_center
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def gen_aug_patch(caseloader,slicei):
    DEBUG = 0
    ctx, cty = caseloader.loadct(slicei)
    dcmstack = caseloader.loadstack(slicei,'dcm')
    dcmimg = dcmstack[:,:,dcmstack.shape[2]//2]

    cartcont = caseloader.load_cart_cont(slicei)

    #cart_patch = croppatch(dcmstack,cty,ctx,256,256)
    #plt.imshow(cart_patch)
    #plt.show()

    mindist = caseloader.load_cart_min_dist(slicei)
    dist_pred_ct_gt, nms_dist_map_p_gt = find_nms_center(mindist,fig=1)

    validy, validx = np.where(nms_dist_map_p_gt[:,:,0]>128)
    '''
    for rd in range(5):
        rnd = random.randint(0,validx.shape[0]-1)
        ckey = '%d-%d'%(validx[rnd],validy[rnd])
        dist_pred_ct_gt[ckey] = nms_dist_map_p_gt[validy[rnd],validx[rnd],0]/256
    '''
    xypos = [[validx[i],validy[i]] for i in range(len(validx))]
    kmeans = KMeans(n_clusters=5, random_state=0).fit(xypos)
    for kmcti in kmeans.cluster_centers_:
        kmcx = int(round(kmcti[0]))
        kmcy = int(round(kmcti[1]))
        ckey = '%d-%d'%(kmcx,kmcy)
        if nms_dist_map_p_gt[kmcy,kmcx,0]<127:
            print('kmean cluster center not large enough',kmcti,nms_dist_map_p_gt[kmcy,kmcx,0])
        dist_pred_ct_gt[ckey] = nms_dist_map_p_gt[kmcy,kmcx,0]/256
    #print(dist_pred_ct_gt)

    aug_patch_obj = {}

    targetdir = DATADESKTOPdir + '/DVWIMAGES/'

    aug_cart_patch_batch = []
    new_cart_patch_batch_filename = '/casepatch/Carotid/'+caseloader.pjname+'/augcart/'+caseloader.pi+os.path.basename(caseloader.dcmpath(slicei))[:-4]+\
                caseloader.side+'.npy'
    aug_patch_obj['cartpatchbatchname'] = new_cart_patch_batch_filename
    if not os.path.exists(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augcart/'):
        os.mkdir(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augcart/')

    aug_polar_patch_batch = []
    new_polar_patch_batch_filename = '/casepatch/Carotid/'+caseloader.pjname+'/augpolar/'+caseloader.pi+os.path.basename(caseloader.dcmpath(slicei))[:-4]+\
                caseloader.side+'.npy'
    aug_patch_obj['polarpatchbatchname'] = new_polar_patch_batch_filename
    if not os.path.exists(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augpolar/'):
        os.mkdir(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augpolar/')

    aug_polar_cont_batch = []
    new_polar_cont_batch_filename = '/casepatch/Carotid/'+caseloader.pjname+'/augpolarcont/'+caseloader.pi+os.path.basename(caseloader.dcmpath(slicei))[:-4]+\
                caseloader.side+'.npy'
    aug_patch_obj['polarcoutbatchname'] = new_polar_cont_batch_filename
    if not os.path.exists(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augpolarcont/'):
        os.mkdir(targetdir+'/casepatch/Carotid/'+caseloader.pjname+'/augpolarcont/')

    aug_patch_obj['auginfo'] = []

    for cti in dist_pred_ct_gt.keys():
        cctx = int(round(int(cti.split('-')[0])+ctx-256))
        ccty = int(round(int(cti.split('-')[1])+cty-256))
        #print(cctx,ccty)
        tx = cctx-ctx
        ty = ccty-cty
        new_cart_patch = croppatch(dcmstack,ccty,cctx,256,256)
        new_cart_patch = new_cart_patch/np.max(new_cart_patch)
        aug_cart_patch_batch.append(new_cart_patch)

        new_polar_patch = topolar(new_cart_patch,256,256)
        new_polar_patch = new_polar_patch/np.max(new_polar_patch)
        #plt.imshow(new_polar_patch)
        #plt.show()
        aug_polar_patch_batch.append(new_polar_patch)

        #rebase polar cont
        new_polar_cont_lumen = tocordpolar(cartcont[0],cctx,ccty)
        new_polar_cont_wall = tocordpolar(cartcont[1],cctx,ccty)
        aug_polar_cont = np.zeros((256,2))
        aug_polar_cont[:,0] = new_polar_cont_lumen
        aug_polar_cont[:,1] = new_polar_cont_wall
        #exportpolarcontour(new_polar_cont_filename,[new_polar_cont_lumen,new_polar_cont_wall])
        aug_polar_cont_batch.append(aug_polar_cont)

        if DEBUG:
            #check polar ct matches with cart patch
            cart_patch_dsp = np.zeros((new_cart_patch.shape[0],new_cart_patch.shape[1],3))
            cart_patch_dsp[:,:,0] = new_cart_patch[:,:,new_cart_patch.shape[2]//2]
            polarbd = np.concatenate([new_polar_cont_lumen[:,None],new_polar_cont_wall[:,None]],axis=1)*256
            contourin,contourout = toctbd(polarbd,256,256)
            cart_vw_seg = plotct(512,contourin,contourout)
            plt.imshow(cart_vw_seg)
            cart_patch_dsp[:,:,1] = cart_vw_seg
            plt.imshow(cart_patch_dsp)
            plt.show()

        augobj = {}
        augobj['ctofx'] = cctx
        augobj['ctofy'] = ccty
        augobj['transx'] = tx
        augobj['transy'] = ty
        augobj['augid'] = len(aug_cart_patch_batch)-1

        aug_patch_obj['auginfo'].append(copy.copy(augobj))
        #print(augobj)

    np.save(targetdir+new_cart_patch_batch_filename,np.array(aug_cart_patch_batch,dtype=np.float16))
    np.save(targetdir+new_polar_patch_batch_filename,np.array(aug_polar_patch_batch,dtype=np.float16))
    np.save(targetdir+new_polar_cont_batch_filename,np.array(aug_polar_cont_batch,dtype=np.float16))

    caseloader.caselist['slices'][slicei]['augpatch'] = aug_patch_obj

    with open(caseloader.caselistname,'wb') as fp:
        pickle.dump(caseloader.caselist,fp)

