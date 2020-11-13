import os
import pickle
import numpy as np
import glob
import pydicom
import random
import cv2
from PolarVW.variables import DATADESKTOPdir
from PolarVW.BB import BB
import matplotlib.pyplot as plt
from PolarVW.UTL import croppatch

class DBLoader():
	def __init__(self,pilist=None):
		if pilist is None:
			self.pilist = {}
		elif type(pilist)==dict:
			self.pilist = pilist
		else:
			print('pilist not a dict')
			self.pilist = {}
		
		self.trainlist = []
		self.vallist = []
		self.testlist = []

	def __repr__(self): 
		msg = 'dbloader with %d pi cases and %d exams.'%(self.size,self.exams)
		if len(self.trainlist)!=0:
			msg += ' Train set %d cases,'%(len(self.trainlist))
		if len(self.vallist)!=0:
			msg += ' Val set %d cases,'%(len(self.vallist))
		if len(self.testlist)!=0:
			msg += ' Test set %d cases.'%(len(self.testlist))
		return msg

	@property
	def size(self):
		return len(self.pilist)

	@property
	def exams(self):
		eict = 0
		for pi in self.pilist:
			eict += len(pi)
		return eict

	def addcase(self,pi,listname):
		if pi in self.pilist:
			self.pilist[pi].append(listname)
		else:
			self.pilist[pi] = [listname]

	def loadsep(self,dbsep):
		self.trainlist = []
		self.vallist = []
		self.testlist = []
		#valid existence in pilist
		for pi in dbsep['train']:
			if pi in self.pilist:
				self.trainlist.append(pi)
		for pi in dbsep['val']:
			if pi in self.pilist:
				self.vallist.append(pi)
		for pi in dbsep['test']:
			if pi in self.pilist:
				self.testlist.append(pi)

	def list_exams(self,dbtype,shuffle=True,sellist=None):
		sel_list = []
		if dbtype == 'train':
			if sellist is None:
				sel_list = self.trainlist
			else:
				for pi in sellist:
					if pi in self.trainlist:
						sel_list.append(pi)
		elif dbtype == 'val':
			if sellist is None:
				sel_list = self.vallist
			else:
				for pi in sellist:
					if pi in self.vallist:
						sel_list.append(pi)
		elif dbtype == 'test':
			if sellist is None:
				sel_list = self.testlist
			else:
				for pi in sellist:
					if pi in self.testlist:
						sel_list.append(pi)
		elif dbtype == 'all':
			if sellist is None:
				sel_list = list(self.pilist.keys())
			else:
				for pi in sellist:
					sel_list.append(pi)
		else:
			print('no such dbtype', dbtype)
			return
		if len(sel_list) == 0:
			print('no case in ', dbtype)
			return
		if shuffle == True:
			random.shuffle(sel_list)
		exams = []
		for pi in sel_list:
			for ei in self.pilist[pi]:
				exams.append(ei)
		return exams

	def listslices(self,dbtype, shuffle=True, sellist=None):
		exams = self.list_exams(dbtype, shuffle,sellist)
		slices = []
		for ei in exams:
			caseloader = CaseLoader(ei)
			slices.extend(caseloader.slices)
		return slices

	def case_generator(self,dbtype,shuffle=True,sellist=None):
		exams = self.list_exams(dbtype,shuffle,sellist)
		while 1:
			for ei in exams:
				caseloader = CaseLoader(ei)
				yield caseloader
			print('all',len(exams),'case loaded')


# def locfilename(filename):
# 	return filename.replace('D:/VW IMAGES/',DATADESKTOPdir+'/DVWIMAGES/').replace('\\','/')

	
class CaseLoader():
	def __init__(self,caselistname):
		assert os.path.exists(caselistname)
		self.caselistname = caselistname
		with open(caselistname,'rb') as fp:
			self.caselist = pickle.load(fp)
		if 'pjname' in self.caselist:
			self.pjname = self.caselist['pjname']
		else:
			self.pjname = self.caselist['QVJfilename'].replace('\\','/').split('/')[-8]
		self.pi = self.caselist['pi']
		self.ei = self.caselist['ei']
		self.side = self.caselist['side']
		if 'art' in self.caselist:
			self.art = self.caselist['art']
		else:
			if self.caselist['pjname']=='knee_artery_analysis':
				self.art = 'Knee'
			elif self.caselist['pjname'] in ['WALLIV0.5mm','reaml_2tp']:
				self.art = 'ICA'
			else:
				self.art = 'Carotid'
		self.targetprefix = DATADESKTOPdir+'/DVWIMAGES/'
		self.targetdir = self.targetprefix+'casepatch/'+self.art+'/'+self.pjname+'/'
		if 'dcmprefix' in self.caselist:
			self.dcmprefix = self.caselist['dcmprefix']
		else:
			self.dcmprefix = ''

	def __repr__(self): 
		return 'project:'+self.pjname+',pi:'+self.caselist['pi']+',ei:'+self.caselist['ei']+',side:'+self.caselist['side']+',slices:'+str(self.size)

	def __getitem__(self, slicei):
		return self.caselist['slices'][slicei]

	def __len__(self):
		return self.size

	@property
	def size(self):
		return len(self.caselist['slices'])
	
	@property
	def slices(self):
		return list(self.caselist['slices'].keys())

	@property
	def examname(self):
		return self.pjname+'-'+self.pi+self.ei+self.side

	def slicename(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		sliceid = self.caselist['slices'][slicei]['sid']['S101']
		return self.pjname+'-'+self.pi+self.ei+'I%d'%sliceid+self.side

	def dcmpath(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		return self.dcmprefix+self.caselist['slices'][slicei]['patch']['dcmname']

	def loaddcm(self,slicei,seq='S101',SCALE=4):
		if not self.valid_slicei(slicei):
			return None
		dcmpath = self.dcmpath(slicei)
		if not os.path.exists(dcmpath):
			dcmpath = self.caselist['dcmprefix'] + dcmpath
		RefDs = pydicom.read_file(dcmpath)
		if RefDs is None:
			return None
		if RefDs.BitsAllocated != 16:
			print('Warning!Bits', RefDs.BitsAllocated, 'SamplesPerPixel', RefDs.SamplesPerPixel)
			RefDs.BitsAllocated = 16
			RefDs.SamplesPerPixel = 1
		RefDs = RefDs.pixel_array

		if RefDs.shape != (512, 512):
			# print('Not regular size',RefDs.shape)
			if RefDs.shape[0] == RefDs.shape[1]:
				RefDs = cv2.resize(RefDs, (512 * SCALE, 512 * SCALE))
			else:
				padarr = np.zeros((max(RefDs.shape), max(RefDs.shape)))
				padarr[0:RefDs.shape[0], 0:RefDs.shape[1]] = RefDs
				RefDs = padarr
				RefDs = cv2.resize(RefDs, (512 * SCALE, 512 * SCALE))
		else:
			RefDs = cv2.resize(RefDs, (512 * SCALE, 512 * SCALE))
		return RefDs

	def loadct(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		return self.caselist['slices'][slicei]['patch']['ctofx'],self.caselist['slices'][slicei]['patch']['ctofy']

	def readct(self,ctfilename):
		conts = []
		with open(ctfilename,'r') as fp:
			line = fp.readline() 
			while line:
				conts.append([float(cti) for cti in line[:-1].split(' ')])
				line = fp.readline()
		return np.array(conts)

	def listiq(self):
		for slicei in self.slices:
			print('slicei location',slicei,'IQ',self.caselist['slices'][slicei]['IQ'])

	def valid_slicei(self,slicei):
		if slicei not in self.caselist['slices']:
			return False
		if 'patch' not in self.caselist['slices'][slicei]:
			return False
		if self.caselist['slices'][slicei]['IQ'] in [-1,0,1]:
			return False
		return True

	def load_polar_cont(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		polar_cont_filename = self.targetprefix+self.caselist['slices'][slicei]['patch']['polarcoutname']
		assert os.path.exists(polar_cont_filename)
		polarcont = self.readct(polar_cont_filename)
		if offset > 0:
			patchheight = polarcont.shape[0]
			polarcontoff = np.zeros(polarcont.shape)
			polarcontoff[offset:] = polarcont[:patchheight - offset]
			polarcontoff[:offset] = polarcont[patchheight - offset:]
			polarcont = polarcontoff
		return polarcont

	def load_cart_cont(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		lumenfilename = self.targetprefix+self.caselist['slices'][slicei]['cont'][0]
		wallfilename = self.targetprefix+self.caselist['slices'][slicei]['cont'][1]
		assert os.path.exists(lumenfilename)
		assert os.path.exists(wallfilename)

		lumenconts = self.readct(lumenfilename)
		wallconts = self.readct(wallfilename)
		ofx = self.caselist['slices'][slicei]['patch']['ofx']
		ofy = self.caselist['slices'][slicei]['patch']['ofy']
		'''
		#only for list from desktop2
		for pti in range(lumenconts.shape[0]):
			lumenconts[pti][0] += ofx
			lumenconts[pti][1] += ofy
		for pti in range(wallconts.shape[0]):
			wallconts[pti][0] += ofx
			wallconts[pti][1] += ofy'''
		#print(ofx,ofy)
		return [lumenconts,wallconts]

	def load_cart_patch(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		cart_patch_filename = self.targetprefix+self.caselist['slices'][slicei]['patch']['cartfilename']
		cart_patch_img = np.load(cart_patch_filename)[:,:,0]
		return cart_patch_img.astype(np.float)

	def load_cart_label(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		cart_label_filename = self.targetprefix+self.caselist['slices'][slicei]['patch']['cartfilename']
		cart_label_img = np.load(cart_label_filename)[:,:,1:3]
		return cart_label_img.astype(np.float)

	def load_cart_vw(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		cart_label_img = self.load_cart_label(slicei)
		return cart_label_img[:,:,1]-cart_label_img[:,:,0]

	def load_cart_min_dist_dirs(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		min_dist_filename = DATADESKTOPdir+'/DVWIMAGES/casepatch/'+self.art+'/'+self.pjname+'/dist/'+self.slicename(slicei)+'.npy'
		if not os.path.exists(min_dist_filename):
			print('no exist mindist',DATADESKTOPdir,min_dist_filename)
			return None
		min_dist_img_dirs = np.load(min_dist_filename).astype(np.float)
		return min_dist_img_dirs

	def load_cart_min_dist(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		min_dist_img_dirs = self.load_cart_min_dist_dirs(slicei)
		if min_dist_img_dirs is None:
			print(self,slicei,'no mindist')
			return None
		min_dist_img = np.min(min_dist_img_dirs,axis=2)
		return min_dist_img/np.max(min_dist_img)

	def load_polar_patch(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		polar_label_filename = self.targetprefix+self.caselist['slices'][slicei]['patch']['polarfilename']
		#print(polar_label_filename)
		polar_label_img = np.load(polar_label_filename)[:,:,0]
		if offset>0:
			patchheight = polar_label_img.shape[0]
			polarpatchoff = np.zeros(polar_label_img.shape)
			polarpatchoff[offset:] = polar_label_img[:patchheight-offset]
			polarpatchoff[:offset] = polar_label_img[patchheight-offset:]
			polar_label_img = polarpatchoff
		return polar_label_img.astype(np.float)

	#return (256, 256, 2), channel 0 lumen, channel 1 outer wall
	def load_polar_label(self,slicei,offset=0):
		if not self.valid_slicei(slicei):
			return None
		polar_label_filename = self.targetprefix+self.caselist['slices'][slicei]['patch']['polarfilename']
		#print(polar_label_filename)
		polar_label_img = np.load(polar_label_filename)[:,:,1:3]
		if offset>0:
			patchheight = polar_label_img.shape[0]
			polarpatchoff = np.zeros(polar_label_img.shape)
			polarpatchoff[offset:] = polar_label_img[:patchheight-offset]
			polarpatchoff[:offset] = polar_label_img[patchheight-offset:]
			polar_label_img = polarpatchoff
		return polar_label_img.astype(np.float)

	def load_stack_by_id(self,sliceidx,loadf,nei=1,offset=0):
		if sliceidx>=self.size or sliceidx<0:
			print('over size')
			return
		else:
			slicei = self.slices[sliceidx]
			return self.loadstack(slicei,loadf,nei,offset)

	def loadstack(self,slicei,loadf,nei=1):
		if not self.valid_slicei(slicei):
			return None

		if loadf=='polar_patch':
			loadfunc = self.load_polar_patch
		elif loadf=='polar_label':
			loadfunc = self.load_polar_label
		elif loadf=='polar_cont':
			loadfunc = self.load_polar_cont
		elif loadf == 'cart_patch':
			loadfunc = self.load_cart_patch
		elif loadf=='cart_label':
			loadfunc = self.load_cart_label
		elif loadf=='cart_cont':
			loadfunc = self.load_cart_cont
		elif loadf=='dcm':
			loadfunc = self.loaddcm
		else:
			print('unknown loadfunc',loadf)
			return

		imgslice = loadfunc(slicei)
		imgstack = np.repeat(np.expand_dims(imgslice,-1),2*nei+1,axis=-1)
		for ni in range(1,nei+1):
			imgslicep = loadfunc(slicei-ni)
			if imgslicep is not None:
				imgstack[...,nei-ni] = imgslicep
			imgslicen = loadfunc(slicei+ni)
			if imgslicen is not None:
				imgstack[...,nei+ni] = imgslicen
		return imgstack

	#aug patch has three neighboring slices
	def load_aug_patch(self,slicei,type):
		if not self.valid_slicei(slicei):
			return None
		if type=='polar_patch':
			aug_filename = self.caselist['slices'][slicei]['augpatch']['polarpatchbatchname']
		elif type == 'cart_patch':
			aug_filename = self.caselist['slices'][slicei]['augpatch']['cartpatchbatchname']
		elif type=='polar_cont':
			aug_filename = self.caselist['slices'][slicei]['augpatch']['polarcoutbatchname']
		elif type == 'cart_label':
			aug_offs = self.load_aug_off(slicei)
			cart_label_vw = self.load_cart_vw(slicei)
			cart_label_vw_aug_batch = []
			for augi in range(len(aug_offs)):
				ctx = aug_offs[augi][0]
				cty = aug_offs[augi][1]
				cart_label_vw_aug = croppatch(cart_label_vw, 256 + cty, 256 + ctx, 256, 256)
				cart_label_vw_aug_batch.append(cart_label_vw_aug)
			return cart_label_vw_aug_batch
		else:
			print('unknown type')
			return
		aug_filename = self.targetprefix + aug_filename
		if not os.path.exists(aug_filename):
			print('no aug polar patch file', aug_filename)
			raise FileNotFoundError
		aug_batch = np.load(aug_filename).astype(np.float)

		return aug_batch

	def load_aug_off(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		aug_offs = []
		augobjs = self.caselist['slices'][slicei]['augpatch']['auginfo']
		for augobji in augobjs:
			if 'ctofx' not in augobji:
				continue
			ofx = augobji['transx']
			ofy = augobji['transy']
			aug_offs.append([ofx,ofy])
		return aug_offs

	def load_gt_tracklets(self):
		if len(list(self.caselist['slices'].keys()))==0:
			print('No slices')
			return None
		maxslicei = np.max(list(self.caselist['slices'].keys()))
		maxsid = self.caselist['slices'][maxslicei]['sid']['S101']
		gttracklet = [[] for i in range(maxsid)]
		for slicei in self.caselist['slices']:
			if not self.valid_slicei(slicei):
				continue
			csid = self.getsid(slicei)
			gtbb = self.load_gt_bb(slicei)
			gtbb.c = 1
			gttracklet[csid].append(gtbb)
		return gttracklet

	def load_gt_bb(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		ctx = self.caselist['slices'][slicei]['patch']['ctofx']/4
		cty = self.caselist['slices'][slicei]['patch']['ctofy']/4
		w = self.load_polar_cont(slicei)[0][1] * 64 + self.load_polar_cont(slicei)[128][1] * 64
		h = self.load_polar_cont(slicei)[64][1] * 64 + self.load_polar_cont(slicei)[192][1] * 64

		return BB(ctx,cty,w,h)

	def getsid(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		return self.caselist['slices'][slicei]['sid']['S101'] - 1

	def plot_polar_ct(self,slicei):
		if not self.valid_slicei(slicei):
			return None
		polarct = self.load_polar_cont(slicei) * 256
		plt.figure(figsize=(5, 5))
		plt.xlim([0, 256])
		plt.ylim([256, 0])
		plt.plot(polarct[::4, 0], np.arange(0, 256, 4), 'o', markersize=2, label='Lumen')
		plt.plot(polarct[::4, 1], np.arange(0, 256, 4), 'o', markersize=2, label='Wall')
		plt.legend()
		plt.show()

