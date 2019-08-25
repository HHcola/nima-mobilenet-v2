# -*- coding: utf-8 -*
import argparse
import os

import sys

sys.path.append(os.path.abspath("../"))
from keras.applications import MobileNetV2
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot
import tensorflow as tf

from utils.utils import ensure_dir_exists
from handlers.data_loader import load_tid_data, load_ava_data
from handlers.loss_fun import earth_mover_loss_tanh, earth_mover_loss
from handlers.evaluation import pearson_correlation, spearman_corr
from callback.tensorboardbatch import TensorBoardBatch
from handlers.data_generator import TrainDataGenerator, val_generator, train_generator

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''


def train(train_image_paths,
          train_image_scores,
          val_image_paths,
          val_image_scores,
          image_size,
          weights_name,
          batchsize,
          epochs,
          steps_epoch,
          val_steps,
          loss_ftype):
    if loss_ftype == 'med':
        loss_fun = earth_mover_loss
        base_model = MobileNet((image_size, image_size, 3), alpha=1.0, include_top=False, pooling='avg')
    else:
        loss_fun = earth_mover_loss_tanh
        base_model = MobileNetV2((image_size, image_size, 3), alpha=1.0, include_top=False, pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    # Dropout
    # rate：0-1
    x = Dropout(0.75)(base_model.output)

    # Dense
    # units：
    # activation：
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.summary()
    # 优化器
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer, loss=loss_fun, metrics=[pearson_correlation, spearman_corr])
    # tensorflow variables need to be initialized before calling model.fit()
    # there is also a tf.global_variables_initializer(); that one doesn't seem to do the trick
    # You will still get a FailedPreconditionError
    K.get_session().run(tf.local_variables_initializer())

    # load weights from trained model if it exists
    if os.path.exists(weights_name):
        model.load_weights(weights_name)

    checkpoint = ModelCheckpoint(weights_name, monitor='val_loss', verbose=1, save_weights_only=True,
                                 save_best_only=True,
                                 mode='min')
    tensorboard = TensorBoardBatch()
    callbacks = [checkpoint, tensorboard]

    training_generator = train_generator(batchsize,
                                         train_image_paths,
                                         train_image_scores)

    # steps_per_epoch
    # validation_steps=ceil(val_dataset_size/batch_size),
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps_epoch,
                                  epochs=epochs, verbose=1, callbacks=callbacks,
                                  validation_data=val_generator(batchsize=batchsize, image_paths=val_image_paths,
                                                                image_scores=val_image_scores),
                                  validation_steps=val_steps)

    # plot metrics
    pyplot.plot(history.history['pearson_correlation'])
    pyplot.savefig(loss_ftype + '_pearson.png')

    pyplot.plot(history.history['spearman_corr'])
    pyplot.savefig(loss_ftype + '_spearman')

    pyplot.plot(history.history['loss_fun'])
    pyplot.savefig(loss_ftype + 'loss_fun.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--data-type', help='train data type:ava,tid', required=False)
    parser.add_argument('-l', '--loss-type', help='loss function type:med,medt', required=False)

    args = parser.parse_args()

    data_type = args.__dict__['data_type']
    loss_fun_type = args.__dict__['loss_type']
    ava_images_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/images/images/'
    ava_score_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/AVA.txt'

    tip_images_path = r'/home/cola/work/nenet/nima/images-data/tid2013/distorted_images/'
    tip_score_path = r'/home/cola/work/nenet/nima/images-data/tid2013/TID2013/tid_labels_train.json'

    if data_type == 'ava':
        ensure_dir_exists(ava_images_path)
        ensure_dir_exists(ava_score_path)
        X, Y = load_ava_data(ava_images_path, ava_score_path)
    else:
        ensure_dir_exists(tip_images_path)
        ensure_dir_exists(tip_score_path)
        X, Y = load_tid_data(tip_images_path, tip_score_path)

    if loss_fun_type == 'med':
        loss_type = 'med'
    else:
        loss_type = 'medt'

    weights_type_name = '../weights/mobilenet_v2_' + data_type + '_' + loss_type + '_weights.h5'

    train_images = X[:-5000]
    train_scores = Y[:-5000]
    val_images = X[-5000:]
    val_scores = Y[-5000:]

    batch_size = 200
    epoch_size = 20
    steps_per_epoch = (train_images.size // batch_size)
    val_steps_epoch = (5000. // batch_size)

    print('loss_fun_type = %s, weights_type_name = %s' % loss_fun_type, weights_type_name)

    train(train_image_paths=train_images,
          train_image_scores=train_scores,
          val_image_paths=val_images,
          val_image_scores=val_scores,
          image_size=224,
          weights_name=weights_type_name,
          batchsize=batch_size,
          epochs=epoch_size,
          steps_epoch=steps_per_epoch,
          val_steps=val_steps_epoch,
          loss_ftype=loss_type
          )
