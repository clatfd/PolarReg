import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import keras
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
import os
from PolarVW.loader import DBLoader,CaseLoader

taskname = 'PolarReg1-1'

if platform == 'win32':
    DESKTOPdir = '//Desktop4'
    DATADESKTOPdir = '//Desktop2'
else:
    # ubuntu
    DESKTOPdir = '/mnt/desktop4'
    DATADESKTOPdir = 'mnt/desktop2'

taskdir = os.path.join(DESKTOPdir, 'Dtensorflow/LiChen/VW/')
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
    dbloader.adddb(dbname, patchdir)

cfg = {}
cfg['width'] = 256
cfg['height'] = 256
cfg['depth'] = 3
cfg['channel'] = 1
cfg['batch_size'] = 512
cfg['G'] = len(get_available_gpus())
