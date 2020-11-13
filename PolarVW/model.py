import keras
from keras.layers import Input, Dense, Conv3D, Conv2D, Reshape, MaxPooling3D, MaxPooling2D, UpSampling3D, UpSampling2D, BatchNormalization, Flatten, Dropout, concatenate
from keras.models import Model

def buildmodel(config):
    #pred img and cont at the same time
    input_img = Input(shape=(config['height'], config['width'], config['depth'], config['channel']), name='patchimg')
    #input_pos = Input(shape=(2,), name='origin')

    af = 'relu'
    kif = 'glorot_normal'

    '''conv3a_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3a_1')(input_img)
    conv3a_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3a_2')(conv3a_1)
    conv3a_2 = BatchNormalization()(conv3a_2)
    mp3a = MaxPooling3D((2, 2, 1), padding='same', name='pooling3a')(conv3a_2)'''

    conv1_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con1_1')(input_img)
    conv1_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    mp1 = MaxPooling3D((2, 2, 1), padding='same', name='pooling1')(conv1_2)
    conv2_1 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con2_1')(mp1)
    conv2_2 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    mp2 = MaxPooling3D((2, 2, 1), padding='same', name='pooling2')(conv2_2)
    conv3_1 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3_1')(mp2)
    conv3_2 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    mp3 = MaxPooling3D((2, 2, 1), padding='same', name='pooling3')(conv3_2)


    conv4_1 = Conv3D(256, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con4_1')(mp3)


    convr1_1 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr1_1')(conv4_1)
    #convr1_2 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr1_2')(convr1_1)
    convr1_2 = BatchNormalization()(convr1_1)
    mp1 = MaxPooling3D((2, 2, 1), padding='same', name='poolingr1')(convr1_2)
    convr2_1 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr2_1')(mp1)
    convr2_2 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr2_2')(convr2_1)
    convr2_2 = BatchNormalization()(convr2_2)
    mp2 = MaxPooling3D((1, 2, 1), padding='same', name='poolingr2')(convr2_2)
    convr3_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr3_1')(mp2)
    convr3_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr3_2')(convr3_1)
    convr3_2 = BatchNormalization()(convr3_2)
    mp3 = MaxPooling3D((1, 2, 1), padding='same', name='poolingr3')(convr3_2)
    convr4_1 = Conv3D(16, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr4_1')(mp3)
    #convr4_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr4_2')(convr4_1)
    regresser = Conv3D(16, (3, 3, 3), activation='sigmoid', padding='same', name='regresser')(convr4_1)
    print(regresser)

    fcn1 = Flatten(name = 'aux_fx1')(regresser)
    fcn1 = Dropout(0.2)(fcn1)
    if 'regnum' in config:
        regnum = config['regnum']
    else:
        regnum = 2
    regr = Dense(config['patchheight']*regnum, name='aux_outputr')(fcn1)
    regr = Reshape((config['patchheight'], regnum), name='reg')(regr)

    if 'gradinput' in config and config['gradinput']:
        input_grad = Input(shape=(config['height'], config['width']),
                          name='patchimg')
        cnn = Model(inputs=[input_grad,input_grad], outputs=regr)
    else:
        cnn = Model(inputs=input_img, outputs=regr)

    #lr=0.01, momentum=0.9,nesterov =True

    return cnn


def Unet3D(config):
    downscale = 4
    inputs = Input(shape=(config['height'], config['width'], config['depth'], config['channel']), name='patchimg')

    conv1 = Conv3D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #conv1 = Conv3D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    conv2 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    #conv2 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    conv3 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    #conv3 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    conv4 = Conv3D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    #conv4 = Conv3D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)

    conv5 = Conv3D(1024//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    #conv5 = Conv3D(1024//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 1))(drop5))
    merge6 = concatenate([drop4, up6], axis=4)
    conv6 = Conv3D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #conv6 = Conv3D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(256//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 1))(conv6))
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    #conv7 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(128//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 1))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    #conv8 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(64//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 1))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    #conv9 = Conv3D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = MaxPooling3D(pool_size=(1, 1, config['depth']))(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def Unet2D(config):
    downscale = 4
    inputs = Input(shape=(config['height'], config['width'], config['channel']), name='patchimg')

    conv1 = Conv2D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64//downscale, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy')

    # model.summary()
    return model

import resnet
def resnetmodel(config):
    patchheight, patchwidth, depth, channel = config['height'], config['width'], config['depth'], config['channel']
    resnet101 = resnet.ResNet101()#weights='imagenet'
    input_img = Input(shape=(patchheight, patchwidth, depth, channel), name='patchimg')
    input_imgr = Reshape((patchheight, patchwidth, depth), name='reshapeinput')(input_img)
    fcn1 = resnet101(input_imgr)

    fcn1 = Dropout(0.2)(fcn1)
    regr = Dense(patchheight * 2, name='aux_outputr')(fcn1)
    regr = Reshape((patchheight, 2), name='reg')(regr)

    cnn = Model(inputs=input_img, outputs=regr)
    return cnn

def simpleCNNModel(config):
    input_img = Input(shape=(config['height'], config['width'], config['depth'], config['channel']), name='patchimg')
    downscale = 4
    conv1 = Conv3D(64//downscale, 3, strides=(2, 2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
    conv1 = Conv3D(64//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    conv2 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    conv3 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    #conv3 = Conv3D(256//downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    conv4 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    #conv4 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)
    conv5 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #conv5 = Conv3D(256 // downscale, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 1))(conv5)
    fcn1 = Flatten(name='fn1')(pool5)
    fcn1 = Dropout(0.2)(fcn1)
    fcn2 = Dense(32, name='fn2',activation='relu')(fcn1)
    fcn2 = Dropout(0.2)(fcn2)
    fcn3 = Dense(1, name='fn3',activation='sigmoid')(fcn2)
    cnn = Model(inputs=input_img, outputs=fcn3)

    return cnn

from keras.applications import VGG16
def vgg16Model(config):
    input_img = Input(shape=(config['height'], config['width'], config['channel']), name='patchimg')
    vgg = VGG16(include_top=False,
        weights="imagenet",
        input_tensor=input_img,
        input_shape=None,
        pooling=None,
        classes=1)
    return vgg
