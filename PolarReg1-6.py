import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import tensorflow as tf
import keras
'''from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.0
set_session(tf.Session(config=config))'''
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

from sys import platform
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus(),platform,tf.__version__,keras.__version__)

import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import cv2
import random
import glob
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,TensorBoard,LambdaCallback
from keras.utils import multi_gpu_model

import sys
sys.path.append(r'\\DESKTOP4\Dtensorflow\LiChen\VW\PolarReg')
from src.loader import DBLoader,CaseLoader
from src.variables import DESKTOPdir,DATADESKTOPdir,MODELDESKTOPdir,taskdir
from src.db import adddb

taskname = 'PolarReg1-6-2'

if not os.path.exists(taskdir + '/' + taskname):
    os.mkdir(taskdir + '/' + taskname)
    print('create new folder ' + taskdir + '/' + taskname)

patchdir = os.path.join(DATADESKTOPdir, 'DVWIMAGES/casepatch/Carotid')
dbnames = ['capricebaseline',
           'careiicarotid',
           'careiicarotidcbir',
           'kowa']
dbloader = DBLoader()
for dbname in dbnames:
    adddb(dbloader, dbname)

cfg = {}
cfg['width'] = 512//2
cfg['height'] = 512//2
cfg['patchheight'] = 256
cfg['depth'] = 3
cfg['channel'] = 1
cfg['G'] = len(get_available_gpus())
cfg['taskdir'] = taskdir
cfg['taskname'] = taskname
cfg['batchsize'] = 32
cfg['rottimes'] = 4
cfg['startepoch'] = 0

# load separated list
with open(taskdir + '/dbsep1174.pickle', 'rb') as fp:
    dbsep = pickle.load(fp)
# dbsep is a dict with 'train':['casename-pi'],'val','test'
dbloader.loadsep(dbsep)

from src.model import buildmodel
from keras.models import load_model
modelname = None
models = glob.glob(os.path.join(taskdir,taskname,'Epo*-*-*.hdf5'))
if len(models) > 0:
    epochs = [float(os.path.basename(i).split('-')[0][3:]) for i in models]
    idxmax = np.argmax(epochs)
    modelname = os.path.basename(models[idxmax])
    print('Most recent', modelname)
    cfg['startepoch'] = int(np.max(epochs))

if cfg['G']==1:
    cnn = buildmodel(cfg)
    if modelname is not None:
        # -------------- load the saved model --------------
        cnn.load_weights(taskdir + '/' + taskname + '/' + modelname)
        print('loaded G=1', modelname)
    #cnn.summary()
else:
    with tf.device('/cpu:0'):
        # initialize the model
        s_cnn = buildmodel(cfg)
        if modelname is not None:
            s_cnn.load_weights(taskdir + '/' + taskname + "/" + modelname)
            print("loaded G=",cfg['G'],modelname)

        s_cnn.load_weights(taskdir + '/PolarReg1-5/Epo242-0.00623-0.83290.hdf5')
        print("loaded G=", cfg['G'], 'PolarReg1-5/Epo242-0.00623-0.83290.hdf5')

        cnn = multi_gpu_model(s_cnn, gpus=cfg['G'])

    # make the model parallel
    cnn = multi_gpu_model(s_cnn, gpus=cfg['G'])
    #s_cnn.summary()



cnn.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss='mae')#dice_coef_loss
print('cnn test',cnn.predict(np.zeros((cfg['G'],cfg['height'],cfg['width'],cfg['depth'],cfg['channel'])))[0].shape)

from src.polarutil import batch_cart_rot

train_exam_ids = dbloader.list_exams('train')
val_exam_ids = dbloader.list_exams('val')

import threading
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):  # Py2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def data_generator(config, exams, aug):
    xarray = np.zeros([config['batchsize'], config['height'], config['width'], config['depth'], config['channel']])
    yarray = np.zeros([config['batchsize'], config['patchheight'], 2])
    bi = 0
    while 1:
        for ei in exams:
            caseloader = CaseLoader(ei)
            # print(caseloader)
            for slicei in caseloader.slices:
                if caseloader.valid_slicei(slicei) == False:
                    continue

                if caseloader.caselist['slices'][slicei]['augpatch'] == [] :
                    continue
                    gen_aug_patch(caseloader, slicei)
                    print(caseloader, slicei, 'gen')

                aug_cart_patch_batch = caseloader.load_aug_patch(slicei, 'cart_patch')
                aug_polar_cont_batch = caseloader.load_aug_patch(slicei, 'polar_cont')
                for augi in range(len(aug_cart_patch_batch)):
                    aug_cart_patch = cv2.resize(aug_cart_patch_batch[augi], (config['width'], config['height']))
                    xarray[bi] = aug_cart_patch[..., None]

                    aug_polar_cont = aug_polar_cont_batch[augi]
                    yarray[bi] = aug_polar_cont_batch[augi]
                    bi += 1
                    if bi == config['batchsize']:
                        bi = 0
                        if aug == True:
                            for offi in range(config['rottimes']):
                                rotsarr = []
                                for i in range(xarray.shape[0]):
                                    rots = np.arange(4)
                                    random.shuffle(rots)
                                    rotsarr.append(rots)
                                rotsarr = np.array(rotsarr)
                                for roti in range(4):
                                    xarray_off, yarray_off = batch_cart_rot(xarray, yarray, rotsarr[:, roti])
                                    yield (xarray_off, yarray_off)
                        else:
                            yield (xarray, yarray)


from src.polarutil import toctbd, polarpredimg, polar_pred_cont_cst, plotct, plotpolar
from src.eval import DSC, diffmap

def get_val_dice(cnn, config):
    val_dsc = []
    taskdir = cfg['taskdir']
    # load array from selected val to test dsc and compare with cart vw label
    if os.path.exists(taskdir + '/valarray_cart.npy'):
        xarray_val = np.load(taskdir + '/valarray_cart.npy')
        xlabel_val = np.load(taskdir + '/vallabel_cart.npy')
    else:
        val_exams = dbloader.list_exams('val', shuffle=False)
        xarray_val = []
        xlabel_val = []
        bi = 0
        for ei in val_exams:
            caseloader = CaseLoader(ei)
            print(caseloader)
            for slicei in caseloader.slices:
                if caseloader.valid_slicei(slicei) == False:
                    continue
                if bi % 20 == 0:
                    polarstack = caseloader.loadstack(slicei, 'cart_patch', nei=config['depth'] // 2)
                    xarray_val.append(polarstack[..., None])
                    cartabel = caseloader.loadstack(slicei, 'cart_label', nei=config['depth'] // 2)[
                        ..., config['depth'] // 2]
                    xlabel_val.append(cartabel[:, :, 1] - cartabel[:, :, 0])
                bi += 1
        xarray_val = np.array(xarray_val)
        xlabel_val = np.array(xlabel_val)
        print(xarray_val.shape, xlabel_val.shape)
        np.save(taskdir + '/valarray_cart.npy', xarray_val)
        np.save(taskdir + '/vallabel_cart.npy', xlabel_val)

    xarray_val_rz = []
    for i in range(xarray_val.shape[0]):
        xarray_val_rz.append(cv2.resize(xarray_val[i, :, :, :, 0], (256, 256)))
    xarray_val_rz = np.array(xarray_val_rz)[..., None]
    polarbd_all = cnn.predict(xarray_val_rz, batch_size=config['G'] * config['batchsize'])
    for ti in range(xarray_val.shape[0]):
        carlabel = xlabel_val[ti]
        # carlabel = cv2.resize(carlabel,(256,256))
        polarimg = xarray_val_rz[ti]
        polarbd = polarbd_all[ti] * config['width']
        # polarprob = polarprobr[:,:,0,1]-polarprobr[:,:,0,0]
        contourin, contourout = toctbd(polarbd, config['height'], config['height'])
        carseg = plotct(512, contourin, contourout)
        cdsc = DSC(carlabel, carseg)
        val_dsc.append(cdsc)

        # pxconf = (np.sum(polarprob[polarseg>0])-np.sum(polarprob[polarseg==0]))/np.sum(polarseg>0)
        # sconf.append(pxconf)

        print('\rPred', ti, '/', xarray_val.shape[0], 'val dice', '%.5f' % np.mean(val_dsc), end="")
    return np.mean(val_dsc)


get_val_dice(cnn,cfg)

class change_training_callback(keras.callbacks.Callback):
    def __init__(self, config):
        self.config = config

    def on_train_begin(self, logs={}):
        print('Train begin')
        taskdir = cfg['taskdir']
        taskname = cfg['taskname']
        self.val_dscs_filename = taskdir + '/' + taskname + '/val_dscs.npy'
        if os.path.exists(self.val_dscs_filename):
            self.val_dscs = np.load(taskdir + '/' + taskname + '/val_dscs.npy').tolist()
        else:
            self.val_dscs = []

    def on_epoch_end(self, epoch, logs={}):
        print('\n=====check results=====at Epoch ', epoch, '\n')
        if epoch % 1 == 0:
            cval_dsc = get_val_dice(cnn,cfg)
            self.val_dscs.append(cval_dsc)
            np.save(self.val_dscs_filename, self.val_dscs)
            tensorboard.val_dscs = self.val_dscs
            taskdir = cfg['taskdir']
            taskname = cfg['taskname']
            if self.config['G'] == 1:
                cnn.save(taskdir + '/' + taskname + '/Epo' + '{:03d}'.format(epoch) + '-' + '{:06.5f}'.format(
                    logs['val_loss']) + '-' + '{:06.5f}'.format(cval_dsc) + '.hdf5')
                print(logs['val_loss'])
            else:
                s_cnn.save_weights(taskdir + '/' + taskname + '/Epo' + '{:03d}'.format(epoch) + '-' + '{:06.5f}'.format(
                    logs['val_loss']) + '-' + '{:06.5f}'.format(cval_dsc) + '.hdf5')
                print(logs['val_loss'])

starttime=datetime.datetime.now()
print('Started',starttime)

ct=change_training_callback(cfg)
logpath = os.path.join(taskdir,taskname,taskname+'_log.csv')
weightspath = os.path.join(taskdir,taskname,"Epo{epoch:03d}-{val_loss:.5f}.hdf5")

'''tensorboard = TensorBoard(log_dir=DESKTOPdir+'/Ftensorflow/LiChen/logs/' + taskname,
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=False)'''

logdir = MODELDESKTOPdir+'/Ftensorflow/LiChen/logs/' + taskname
logdirprof = MODELDESKTOPdir+'/Ftensorflow/LiChen/logs/' + taskname+'/training/plugins/profile/'
if not os.path.exists(logdir):
    os.makedirs(logdirprof)

from src.log import TrainValTensorBoard
tensorboard = TrainValTensorBoard(log_dir=logdir,write_graph=False)


callbacklist = [ct,
               EarlyStopping(patience=100),
               #CSVLogger(logpath, separator=',', append=True),
               tensorboard
               ]
if cfg['G']==1:
    callbacklist.append(ModelCheckpoint(weightspath, monitor='val_loss', period=1, verbose=1, save_best_only=True, mode='min'))


trainsteps = int(round(dbloader.exams*15*cfg['rottimes']//cfg['batchsize']*0.8))//100
valsteps = int(round(dbloader.exams*15//cfg['batchsize']*0.2))//40
print('trainsteps,valsteps',trainsteps,valsteps)

traingen = data_generator(cfg,train_exam_ids,0)
valgen = data_generator(cfg,val_exam_ids,0)

cnn.fit_generator(traingen,
                          steps_per_epoch=trainsteps,
                          epochs=1000,
                          verbose=1,
                          max_queue_size=50,
                          validation_data=valgen,
                          validation_steps=valsteps,
                          initial_epoch=cfg['startepoch'],
                          callbacks=callbacklist
                 )
endtime=datetime.datetime.now()
print('Ended',endtime)
elaspsetime=endtime-starttime
print(divmod(elaspsetime.total_seconds(), 60)[0],'minutes')
if cfg['G']==1:
    cnn.save(taskdir + '/' + taskname + '/' + 'finalmodel.hdf5')
else:
    s_cnn.save_weights(taskdir + '/' + taskname + '/' + 'finalmodel.hdf5')
