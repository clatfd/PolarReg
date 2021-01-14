import os
import numpy as np
import random
from PolarVW.loader import DBLoader
from PolarVW.db import adddb
from PolarVW.variables import cdir

dbnames = ['capricebaseline',
           'careiicarotid',
           'careiicarotidcbir',
           'kowa']
dbloader = DBLoader()
for dbname in dbnames:
    adddb(dbloader,dbname)

tot_size=1157#len(list(set([i[:i.find('E')] for i in case_raw])))
case_unique_filename = cdir+'/case_uniq'+str(tot_size)+'.npy'
if not os.path.exists(case_unique_filename):
    case_uniq = list(set([i['casename'] for i in dbloader.pilist]))
    random.shuffle(case_uniq)
    np.save('case_uniq'+str(tot_size), case_uniq)
    print('generate','case_uniq'+str(tot_size))
case_uniq = np.load(case_unique_filename)
print('loaded', case_unique_filename, case_uniq.shape)

trainsp = int(0.8 * len(case_uniq))
valsp = int(0.9 * len(case_uniq))

train_caselist = case_uniq[:trainsp]
val_caselist = case_uniq[trainsp:valsp]
test_caselist = case_uniq[valsp:]

dbloader.trainlist = train_caselist
dbloader.vallist = val_caselist
dbloader.testlist = test_caselist

casegenerator = dbloader.case_generator('train')


caseloader = next(casegenerator)
for slicei in caseloader.slices:
    print(caseloader.loadstack(slicei,'polar_patch').shape)
    print(caseloader.loadstack(slicei, 'polar_cont').shape)
