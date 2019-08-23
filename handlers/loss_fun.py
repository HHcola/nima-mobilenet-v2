# -*- coding: utf-8 -*
from keras import backend as K


def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


def earth_mover_loss_tanh(y_true, y_pred):
    cdf_ytrue = K.cumsum(K.tanh(y_true), axis=-1)
    cdf_ypred = K.cumsum(K.tanh(y_pred), axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)
