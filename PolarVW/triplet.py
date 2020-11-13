import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
import numpy as np
import os

def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(in_dims[0], in_dims[1], in_dims[2],), activation='relu',
                     name='conv1'))
    model.add(MaxPooling2D((2, 2), None, padding='same', name='pool1'))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D((2, 2), None, padding='same', name='pool2'))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', name='conv3'))
    model.add(MaxPooling2D((2, 2), None, padding='same', name='pool3'))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', name='conv4'))
    model.add(MaxPooling2D((2, 2), None, padding='same', name='pool4'))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', name='conv5'))
    model.add(MaxPooling2D((2, 2), None, padding='same', name='pool5'))
    model.add(Flatten(name='flatten'))
    # model.add(Dense(40,name='embeddings'))
    model.add(Dense(64))

    return model

def create_triplet_model():
    Patchsize = 128
    Channel = 1
    anchor_input = Input((Patchsize, Patchsize, Channel,), name='anchor_input')
    positive_input = Input((Patchsize, Patchsize, Channel,), name='positive_input')
    negative_input = Input((Patchsize, Patchsize, Channel,), name='negative_input')

    # Shared embedding layer for positive and negative items
    Shared_DNN = create_base_network([Patchsize, Patchsize, Channel, ])

    encoded_anchor = Shared_DNN(anchor_input)
    encoded_positive = Shared_DNN(positive_input)
    encoded_negative = Shared_DNN(negative_input)

    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)

    return model

def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss

#if USETrainEpoch:
def triplet_generator(numpyfolder, dblist, batchsize):
    Patchsize = 128
    Channel = 1
    mi = 0
    Anchor = np.zeros((batchsize, Patchsize, Patchsize, Channel))
    Positive = np.zeros((batchsize, Patchsize, Patchsize, Channel))
    Negative = np.zeros((batchsize, Patchsize, Patchsize, Channel))
    Y_dummy = np.empty((batchsize, 1))
    while 1:
        totnum = 0
        for casei in dblist:
            if not os.path.exists(numpyfolder+'/'+casei+'.npy'):
                continue
            X_train = np.load(numpyfolder+'/'+casei+'.npy').astype(np.float)
            for tripi in range(len(X_train)):
                Anchor[mi,:,:,0] = X_train[tripi,0]
                Positive[mi,:,:,0] = X_train[tripi,1]
                Negative[mi,:,:,0] = X_train[tripi,2]
                mi += 1
                totnum += 1
                if mi == batchsize:
                    mi = 0
                    yield ([Anchor, Positive, Negative], Y_dummy)
        print('tot num', totnum)


def load_triplet_pairs(numpyfolder,dblist):
    Anchor = []
    Positive = []
    Negative = []
    for casei in dblist:
        if not os.path.exists(numpyfolder + '/' + casei + '.npy'):
            #print(numpyfolder + '/' + casei + '.npy not exist')
            continue
        X_train = np.load(numpyfolder + '/' + casei + '.npy').astype(np.float)
        for tripi in range(len(X_train)):
            Anchor.append(X_train[tripi, 0])
            Positive.append(X_train[tripi, 1])
            Negative.append(X_train[tripi, 2])
    Y_dummy = np.empty((len(Negative), 1))
    return [np.array(Anchor)[:,:,:,None],np.array(Positive)[:,:,:,None], np.array(Negative)[:,:,:,None]], Y_dummy

def checktriplet(triplet):
    if len(triplet.shape)==4:
        triplet = triplet[:,:,:,0]
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title('Anchor')
    plt.imshow(triplet[0])
    plt.subplot(1,3,2)
    plt.title('Positive')
    plt.imshow(triplet[1])
    plt.subplot(1,3,3)
    plt.title('Negative')
    plt.imshow(triplet[2])
    plt.show()


def evaltest(featuremodel,Anchor_test, Positive_test, Negative_test):
    correct = 0
    wrong = 0

    for ti in range(len(Anchor_test)):
        anchorfeat = featuremodel.predict(Anchor_test[ti:ti+1])
        posfeat = featuremodel.predict(Positive_test[ti:ti+1])
        negfeat = featuremodel.predict(Negative_test[ti:ti+1])

        #print(np.sum(abs(anchorfeat-posfeat)),np.sum(abs(anchorfeat-negfeat)))
        if np.sum(abs(anchorfeat-posfeat))<np.sum(abs(anchorfeat-negfeat)):
            correct += 1
        else:
            wrong += 1

    print(correct,wrong)
    return correct/(correct+wrong)

def bb_match_score_patch(featuremodel, patch1, patch2):
    patchfeat1 = featuremodel.predict(patch1[None,:,:,None])[0]
    patchfeat2 = featuremodel.predict(patch2[None,:,:,None])[0]
    return np.sum(abs(patchfeat1-patchfeat2))

from PolarVW.UTL import croppatch
def bb_match_score(featuremodel, imgstack, pos1, pos2):
    patch1 = croppatch(imgstack[pos1[2]], pos1[1], pos1[0], 64, 64)
    patch2 = croppatch(imgstack[pos2[2]], pos2[1], pos2[0], 64, 64)
    return bb_match_score_patch(featuremodel, patch1, patch2)


def create_class_model():
    Classifier_input1 = Input((512,))
    Classifier_input2 = Input((512,))
    Classifier_input = concatenate([Classifier_input1, Classifier_input2], axis=-1, name='feat_merged_layer')
    Classifier_output = Dense(128, activation='softmax')(Classifier_input)
    Classifier_output = Dense(40, activation='softmax')(Classifier_output)
    Classifier_output = Dense(1)(Classifier_output)
    Classifier_model = Model([Classifier_input1,Classifier_input2], Classifier_output)
    return Classifier_model

def load_class_data(featuremodel, Anchor, Positive, Negative):
    class_anchor_patches_all = []
    class_posneg_patches_all = []
    classes_all = []
    batchsize = 32
    for bi in range(len(Anchor)//batchsize+1):
        #print(bi*batchsize,(bi+1)*batchsize)
        anchor_patch_feat = featuremodel.predict(Anchor[bi*batchsize:(bi+1)*batchsize])
        pos_patch_feat = featuremodel.predict(Positive[bi*batchsize:(bi+1)*batchsize])
        neg_patch_feat = featuremodel.predict(Negative[bi*batchsize:(bi+1)*batchsize])
        class_anchor_patches = np.concatenate([anchor_patch_feat,anchor_patch_feat])
        class_posneg_patches = np.concatenate([pos_patch_feat,neg_patch_feat])
        classes = np.concatenate([np.ones((anchor_patch_feat.shape[0])),np.zeros((anchor_patch_feat.shape[0]))])
        class_anchor_patches_all.extend(class_anchor_patches)
        class_posneg_patches_all.extend(class_posneg_patches)
        classes_all.extend(classes)
    return class_anchor_patches_all, class_posneg_patches_all, classes_all
