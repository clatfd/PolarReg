import tensorflow as tf
from keras import backend as K


def polarloss(y_true,y_pred):
    dmax = K.sum(tf.math.maximum(y_true,y_pred))
    dmin = K.sum(tf.math.minimum(y_true,y_pred))
    return K.log(tf.math.divide(dmax,dmin))
