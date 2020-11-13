from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import keras.backend as K
import numpy as np
import tensorflow as tf
import os
import pydicom
import cv2
from .utils import WeightReader, decode_netout, draw_boxes, normalize
class Yolo:
	def __init__(self,yolomodelname=None):
		#self.yolomodelname = r'D:\tensorflow\LiChen\basic-yolo-keras-master\YoloN1\Rep2weights_023-0.00270.h5.h5'
		#self.yolomodelname = r'D:\tensorflow\LiChen\Yolo\YoloN1\Rep3weights_020-0.00186.h5'
		if yolomodelname is None:
			self.yolomodelname = 'F:/tensorflow/LiChen/Models/Yolo/YoloN1/Rep3weights_020-0.00186.h5'
		else:
			self.yolomodelname = yolomodelname
		self.LABELS = ['Artery']

		self.IMAGE_H, self.IMAGE_W = 384, 384
		GRID_H,  GRID_W  = 12 , 12
		BOX			  = 5
		self.CLASS			= len(self.LABELS)
		CLASS_WEIGHTS	= np.ones(self.CLASS, dtype='float32')
		self.OBJ_THRESHOLD	= 0.3#0.5
		self.NMS_THRESHOLD	= 0.3#0.45
		self.ANCHORS		  = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

		NO_OBJECT_SCALE  = 1.0
		OBJECT_SCALE	 = 5.0
		COORD_SCALE	  = 1.0
		CLASS_SCALE	  = 1.0

		BATCH_SIZE	   = 16
		WARM_UP_BATCHES  = 0
		TRUE_BOX_BUFFER  = 50

		#yolo structure
		input_image = Input(shape=(self.IMAGE_H, self.IMAGE_W, 3))
		true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

		# Layer 1
		x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 2
		x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
		x = BatchNormalization(name='norm_2')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 3
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
		x = BatchNormalization(name='norm_3')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 4
		x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
		x = BatchNormalization(name='norm_4')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 5
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
		x = BatchNormalization(name='norm_5')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 6
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
		x = BatchNormalization(name='norm_6')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 7
		x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
		x = BatchNormalization(name='norm_7')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 8
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
		x = BatchNormalization(name='norm_8')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 9
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
		x = BatchNormalization(name='norm_9')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 10
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
		x = BatchNormalization(name='norm_10')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 11
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
		x = BatchNormalization(name='norm_11')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 12
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
		x = BatchNormalization(name='norm_12')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 13
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
		x = BatchNormalization(name='norm_13')(x)
		x = LeakyReLU(alpha=0.1)(x)

		skip_connection = x

		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 14
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
		x = BatchNormalization(name='norm_14')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 15
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
		x = BatchNormalization(name='norm_15')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 16
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
		x = BatchNormalization(name='norm_16')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 17
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
		x = BatchNormalization(name='norm_17')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 18
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
		x = BatchNormalization(name='norm_18')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 19
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
		x = BatchNormalization(name='norm_19')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 20
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
		x = BatchNormalization(name='norm_20')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 21
		skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
		skip_connection = BatchNormalization(name='norm_21')(skip_connection)
		skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
		skip_connection = Lambda(self._space_to_depth_x2)(skip_connection)

		x = concatenate([skip_connection, x])

		# Layer 22
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
		x = BatchNormalization(name='norm_22')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 23
		x = Conv2D(BOX * (4 + 1 + self.CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
		output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + self.CLASS))(x)
		# small hack to allow true_boxes to be registered when Keras build the model 
		# for more information: https://github.com/fchollet/keras/issues/2790
		output = Lambda(lambda args: args[0])([output, true_boxes])

		self.yolomodel = Model([input_image, true_boxes], output)

		self.yolomodel.load_weights(self.yolomodelname)
		self.dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

	# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
	def _space_to_depth_x2(self,x):
		return tf.space_to_depth(x, block_size=2)

	def predstack(self,dicomstack,EXPORT_PRED=1):
		#check consistency of id incremental (from 001 +=1 to max)
		bbr = [[] for i in range(dicomstack.shape[0])]
		if EXPORT_PRED:
			predimgstack = np.zeros((dicomstack.shape[0],dicomstack.shape[1],dicomstack.shape[2],3),dtype=np.uint8)
		for slicei in range(dicomstack.shape[0]):
			dicomslice = dicomstack[slicei]
			if np.max(dicomslice)==0:
				continue
			if EXPORT_PRED:
				rtbox,predimage = self.predslice(dicomslice,EXPORT_PRED)
				predimgstack[slicei] = predimage
			else:	
				rtbox = self.predslice(dicomslice,EXPORT_PRED)
			if rtbox==0 and np.max(dicomslice)>0:
				print('slicei',slicei,'No Bounding box detected')
				continue
			bbr[slicei].extend(rtbox)
		if EXPORT_PRED:
			return bbr,predimgstack
		else:	
			return bbr

	def predslice(self,dicomslice,EXPORT_PRED=1):
		#normalize to 255
		if np.max(dicomslice)==0:
			return
		maxperc = np.percentile(dicomslice,99.8)
		dicomslice[dicomslice>maxperc] = maxperc
		dcmimgc = dicomslice/np.max(dicomslice)*255
		image = np.repeat(dcmimgc[:,:,None],3,axis=2)
		input_image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
		input_image = input_image / np.max(dcmimgc)
		input_image = np.expand_dims(input_image, 0)

		netout = self.yolomodel.predict([input_image, self.dummy_array])

		boxes = decode_netout(netout[0], 
							  obj_threshold=self.OBJ_THRESHOLD,
							  nms_threshold=self.NMS_THRESHOLD,
							  anchors=self.ANCHORS, 
							  nb_class=self.CLASS)
		
		rtboxes = []

		for selboxi in range(len(boxes)):
			selbox = boxes[selboxi]
			xmin  = (selbox.x - selbox.w/2) * image.shape[1]
			xmax  = (selbox.x + selbox.w/2) * image.shape[1]
			ymin  = (selbox.y - selbox.h/2) * image.shape[0]
			ymax  = (selbox.y + selbox.h/2) * image.shape[0]
			x = selbox.x * image.shape[1]
			y = selbox.y * image.shape[0]
			w = selbox.w * image.shape[1]
			h = selbox.h * image.shape[0]
			classes = selbox.classes.tolist()
			rtbox = [x,y,w,h]+classes
			rtboxes.append(rtbox)
		if EXPORT_PRED == 1:
			#dcmi = os.path.basename(dicomfilename)
			#predname = os.path.join(predpath,dcmi+'_%d_%.2f_%.2f_%.2f_%.2f_.jpg'%(len(boxes),xmin,ymin,xmax,ymax))
			predimage = draw_boxes(image, boxes, labels=self.LABELS)
			#plt.imshow(image[:,:,::-1])
			#plt.show()
			#cv2.imwrite(predname,image.astype(np.uint8))
			return rtboxes,predimage
		else:
			return rtboxes
