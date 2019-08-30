# -*- coding: utf-8 -*
import os
import sys
from shutil import copyfile

sys.path.append(os.path.abspath("../"))

import numpy as np
import argparse
from path import Path
from keras.applications import MobileNetV2
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.score_utils import mean_score, std_score


def evaluate(imgs,
             model_weights_path,
             save_path,
             target_size,
             rank_images,
             loss_type,
             aim_img_save_path
             ):
    """
    evaluate
    :return:
    """
    with tf.device('/CPU:0'):

        if loss_type == 'med':
            base_model = MobileNet((None, None, 3), alpha=1.0, include_top=False, pooling='avg', weights=None)
        else:
            base_model = MobileNetV2((None, None, 3), alpha=1.0, include_top=False, pooling='avg', weights=None)

        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights(model_weights_path)

        score_list = []

        for img_path in imgs:
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)

            file_name = Path(img_path).name.lower()
            score_list.append((file_name, mean, std))

            print("Evaluating : ", img_path)
            print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
            print()

        if rank_images:
            print("*" * 40, "Ranking Images", "*" * 40)
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

            for i, (name, score, std) in enumerate(score_list):
                print("%d)" % (i + 1), "%s : Score = %0.5f, Std = %0.5f" % (name, score, std))

        # save to file
        fo = open(save_path, 'a')
        fo.write('===== begin ====\n')
        for i, (name, score, std) in enumerate(score_list):
            name_score = str(i + 1) + ',' + name + " Score = " + str(score) + " Std = " + str(std)
            fo.write(name_score)
            fo.write('\n')
        fo.write('===== end ====\n')
        fo.close()

        # save image to file
        name = score_list[0][0]
        src = os.path.join(Path(imgs[0]).parent, name)
        dst = os.path.join(aim_img_save_path, name)
        copyfile(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--data-type', help='train data type:ava,tid', required=False)
    parser.add_argument('-l', '--loss-type', help='loss function type:emd,emdt', required=False)

    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it', required=True)

    parser.add_argument('-sf', type=str, default='default',
                        help='save file after they have been scored')

    parser.add_argument('-resize', type=str, default='false',
                        help='Resize images to 224x224 before scoring')

    parser.add_argument('-rank', type=str, default='true',
                        help='Whether to tank the images after they have been scored')

    args = parser.parse_args()

    data_type = args.__dict__['data_type']
    loss_fun_type = args.__dict__['loss_type']

    if loss_fun_type == 'med':
        loss_type = 'med'
    else:
        loss_type = 'medt'

    weights_path = '../weights/mobilenet_v2_' + data_type + '_' + loss_type + '_weights.h5'

    resize_image = args.resize.lower() in ("true", "yes", "t", "1")
    target_image_size = (224, 224) if resize_image else None
    rank_images_score = args.rank.lower() in ("true", "yes", "t", "1")

    file_dir = os.listdir(args.dir)
    dirs = []
    for dir_file in file_dir:
        dir_file_path = os.path.join(args.dir, dir_file)
        if os.path.isdir(dir_file_path):
            dirs.append(dir_file)

    for dir_name in dirs:
        imgs_path = []
        path = os.path.join(args.dir, dir_name)
        if not os.path.exists(path + '/score'):
            os.mkdir(path + '/score')
        save_score_path = path + '/score/' + data_type + '_score'
        imgs_path = Path(path).files('*.png')
        imgs_path += Path(path).files('*.jpg')
        imgs_path += Path(path).files('*.jpeg')

        aim_path = os.path.join(args.dir, "aim")
        if not os.path.exists(aim_path):
            os.mkdir(aim_path)
        aim_img_path = os.path.join(aim_path, dir_name)
        if not os.path.exists(aim_img_path):
            os.mkdir(aim_img_path)
        print("evaluate : args.dir = %s, weights_path = %s, save_score_path = %s",
              args.dir, weights_path, save_score_path)
        evaluate(imgs=imgs_path,
                 model_weights_path=weights_path,
                 save_path=save_score_path,
                 target_size=target_image_size,
                 rank_images=rank_images_score,
                 loss_type=loss_type,
                 aim_img_save_path=aim_img_path
                 )
