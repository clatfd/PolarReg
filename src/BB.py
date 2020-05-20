import numpy as np
import copy

class BB:
	def __init__(self,  x, y, w, h, c = None, classes = None):
		self.x	 = x
		self.y	 = y
		self.w	 = w
		self.h	 = h
		self.c	 = c
		self.classes = classes
		self.label = -1
		self.score = -1

	@classmethod
	def fromminmax(cls, xmin, xmax, ymin, ymax, c = None, classes = None):
		x = (xmin+xmax)/2
		y = (ymin+ymax)/2
		w = xmax-xmin
		h = ymax-ymin
		return cls(x,y,w,h,c,classes)

	@classmethod
	def fromminmaxobj(cls, bbobject, c = None, classes = None):
		xmin = bbobject['xmin']
		xmax = bbobject['xmax']
		ymin = bbobject['ymin']
		ymax = bbobject['ymax']
		x = (xmin+xmax)/2
		y = (ymin+ymax)/2
		w = xmax-xmin
		h = ymax-ymin
		return cls(x,y,w,h,c,classes)

	@classmethod
	def fromlist(cls, xywh, c = None, classes = None):
		assert len(xywh)>=4
		return cls(xywh[0],xywh[1],xywh[2],xywh[3],c,classes)

	@classmethod
	def fromlistclass(cls, xywh, classes = None):
		if classes is None:
			assert len(xywh)>=4
			if len(xywh)>4:
				return cls(xywh[0],xywh[1],xywh[2],xywh[3],np.max(xywh[4:]),xywh[4:])
			else:
				return cls(xywh[0],xywh[1],xywh[2],xywh[3])
		else:
			return cls(xywh[0],xywh[1],xywh[2],xywh[3],np.max(classes),classes)

	@classmethod
	def fromminmaxlist(cls, minmax, c = None, classes = None):
		xmin = minmax[0]
		xmax = minmax[2]
		ymin = minmax[1]
		ymax = minmax[3]
		if len(minmax)==5 and c is None:
			c = minmax[4]
			classes = [c]
		x = (xmin+xmax)/2
		y = (ymin+ymax)/2
		w = xmax-xmin
		h = ymax-ymin
		return cls(x,y,w,h,c,classes)

	@classmethod
	#minmax, confidence, classtype (single int number)
	#classnumber how many classes
	def fromminmaxlistclabel(cls, minmaxct, classnumber):
		xmin = minmaxct[0]
		xmax = minmaxct[2]
		ymin = minmaxct[1]
		ymax = minmaxct[3]
		x = (xmin+xmax)/2
		y = (ymin+ymax)/2
		w = xmax-xmin
		h = ymax-ymin
		c = minmaxct[4]
		cclasses = []
		for i in range(classnumber):
			if i == int(minmaxct[5]):
				cclasses.append(c)
			else:
				cclasses.append(0)
		return cls(x,y,w,h,c,classes=cclasses)

	def __repr__(self): 
		rtstr = 'x:%.2f,y:%.2f,w:%.2f,h:%.2f'%(self.x,self.y,self.w,self.h)
		if self.c is not None:
			rtstr += ',c:%.2f'%(self.c)
		if self.classes is not None:
			rtstr += ',class:'+','.join(str(x) for x in self.classes)
		return rtstr

	def __add__(self,a):
		bbflat1 = self.getbbclassflat()
		bbflat2 = a.getbbclassflat()
		bbflatcom = [bbflat1[i]+bbflat2[i] for i in range(len(bbflat1))]
		return self.fromlistclass(bbflatcom)

	def __mul__(self, other):
		bbflat1 = self.getbbclassflat()
		bbflatcom = [bbflat1[i]*other for i in range(len(bbflat1))]
		return self.fromlistclass(bbflatcom)

	def __truediv__(self, other):
		bbflat1 = self.getbbclassflat()
		bbflatcom = [bbflat1[i]/other for i in range(len(bbflat1))]
		return self.fromlistclass(bbflatcom)

	@property
	def xmin(self):
		return int(round(self.x - self.w / 2))

	@property
	def xmax(self):
		return int(round(self.x + self.w / 2))

	@property
	def ymin(self):
		return int(round(self.y - self.h / 2))

	@property
	def ymax(self):
		return int(round(self.y + self.h / 2))

	def get_score(self):
		assert(self.classes is not None)
		if self.score == -1:
			self.score = self.classes[self.get_label()]
		return self.score

	def get_label(self):
		assert(self.classes is not None)
		if self.label == -1:
			self.label = np.argmax(self.classes)
		return self.label
	
	def getbb(self):
		return [self.x,self.y,self.w,self.h]

	def getbbc(self):
		if self.classes is None:
			return [self.x,self.y,self.w,self.h,0]
		else:		
			return [self.x,self.y,self.w,self.h,self.get_score()]

	def getbbclabel(self):
		if self.classes is None:
			return [self.x,self.y,self.w,self.h,0]
		else:		
			return [self.x,self.y,self.w,self.h,self.get_score(),self.get_label()]

	def getbbclass(self):
		if self.classes is None:
			return [self.x,self.y,self.w,self.h,0]
		else:		
			return [self.x,self.y,self.w,self.h,self.classes]
	
	def getbbclassflat(self):
		if self.classes is None:
			return [self.x,self.y,self.w,self.h,0]
		else:		
			return [self.x,self.y,self.w,self.h]+self.classes

	def getminmax(self):
		xmin  = (self.x - self.w/2) 
		xmax  = (self.x + self.w/2) 
		ymin  = (self.y - self.h/2) 
		ymax  = (self.y + self.h/2) 
		return [xmin,ymin,xmax,ymax]

	def getminmaxclabel(self):
		xmin  = (self.x - self.w/2) 
		xmax  = (self.x + self.w/2) 
		ymin  = (self.y - self.h/2) 
		ymax  = (self.y + self.h/2)
		if self.classes is None:
			return [xmin,ymin,xmax,ymax,self.c]
		else:
			return [xmin,ymin,xmax,ymax,self.get_score(),self.get_label()]

	def get_expand_iou(self,bb2,scale):
		bbr1 = copy.deepcopy(self)
		bbr2 = copy.deepcopy(bb2)
		bbr1.w = bbr1.w*scale
		bbr1.h = bbr1.h*scale
		bbr2.w  = bbr2.w*scale
		bbr2.h = bbr2.h*scale
		return bbr1.get_iou(bbr2)

	def get_iou(self, bb2BB):
		bb1 = self.getminmax()
		bb2 = bb2BB.getminmax()
		"""
		Calculate the Intersection over Union (IoU) of two bounding boxes.

		Parameters
		----------
		bb1 : dict
			Keys: {0, 2, 1, 3}
			The (x1, y1) position is at the top left corner,
			the (x2, y2) position is at the bottom right corner
		bb2 : dict
			Keys: {0, 2, 1, 3}
			The (x, y) position is at the top left corner,
			the (x2, y2) position is at the bottom right corner

		Returns
		-------
		float
			in [0, 1]
		"""
		assert bb1[0] < bb1[2]
		assert bb1[1] < bb1[3]
		assert bb2[0] < bb2[2]
		assert bb2[1] < bb2[3]

		# determine the coordinates of the intersection rectangle
		x_left = max(bb1[0], bb2[0])
		y_top = max(bb1[1], bb2[1])
		x_right = min(bb1[2], bb2[2])
		y_bottom = min(bb1[3], bb2[3])

		if x_right < x_left or y_bottom < y_top:
			return 0.0

		# The intersection of two axis-aligned bounding boxes is always an
		# axis-aligned bounding box
		intersection_area = (x_right - x_left) * (y_bottom - y_top)

		# compute the area of both AABBs
		bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
		bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
		assert iou >= 0.0
		assert iou <= 1.0
		return iou

	def get_iom(self, bb2BB):
		#intersection over larger (A, B). Percent of overlap.
		bb1 = self.getminmax()
		bb2 = bb2BB.getminmax()
		
		assert bb1[0] < bb1[2]
		assert bb1[1] < bb1[3]
		assert bb2[0] < bb2[2]
		assert bb2[1] < bb2[3]

		# determine the coordinates of the intersection rectangle
		x_left = max(bb1[0], bb2[0])
		y_top = max(bb1[1], bb2[1])
		x_right = min(bb1[2], bb2[2])
		y_bottom = min(bb1[3], bb2[3])

		if x_right < x_left or y_bottom < y_top:
			return 0.0

		# The intersection of two axis-aligned bounding boxes is always an
		# axis-aligned bounding box
		intersection_area = (x_right - x_left) * (y_bottom - y_top)

		# compute the area of both AABBs
		bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
		bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iom = max(intersection_area / float(bb1_area), intersection_area / float(bb2_area))
		assert iom >= 0.0
		assert iom <= 1.0
		return iom

	def get_bb_fit(self,bbjr):
		bbii = self.getbb()
		bbj = bbjr.getbb()
		closs = 0
		for i in range(4):
			closs += abs(bbii[i]-bbj[i])
		return closs

	def range_overlap(self,r2r):
		r1 = self.getbb()
		r2 = r2r.getbb()
		return (max(r1[0],r2[0]) <= min(r1[2],r2[2])) and (max(r1[1],r2[1]) <= min(r1[3],r2[3]))

	def insidebb(self,x,y):
		xmin = (self.x - self.w / 2)
		xmax = (self.x + self.w / 2)
		ymin = (self.y - self.h / 2)
		ymax = (self.y + self.h / 2)
		if x>xmax or x<xmin or y>ymax or y<ymin:
			return False
		else:
			return True
