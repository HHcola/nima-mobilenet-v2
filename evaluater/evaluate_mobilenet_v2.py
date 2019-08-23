# -*- coding: utf-8 -*
import numpy as np
import argparse
from path import Path
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from utils.score_utils import mean_score, std_score


parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-resize', type=str, default='false',
                    help='Resize images to 224x224 before scoring')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to tank the images after they have been scored')

parser.add_argument('-sf', type=str, default='default',
                    help='save file after they have been scored')

parser.add_argument('-weight', type=str, default='merge',
                    help='Weight:merge(weight),ava_weight,tid_weight')

args = parser.parse_args()
resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (224, 224) if resize_image else None
rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = Path(args.dir).files('*.png')
    imgs += Path(args.dir).files('*.jpg')
    imgs += Path(args.dir).files('*.jpeg')

elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')


WEIGHT_TYPE_MERGE = 'merge'
WEIGHT_TYPE_AVA = 'ava_weight'
WEIGHT_TYPE_TID = 'tid_weight'

weight_type = 'merge'
if args.weight is not None:
    weight_type = args.weight


if weight_type == WEIGHT_TYPE_MERGE:
    model_weights_path = 'weights/mobilenet_v2_weights.h5'
elif weight_type == WEIGHT_TYPE_AVA:
    model_weights_path = 'weights/mobilenet_v2_ava_weights.h5'
elif weight_type == WEIGHT_TYPE_TID:
    model_weights_path = 'weights/mobilenet_v2_tid_weights.h5'
else:
    model_weights_path = 'weights/mobilenet_v2_weights.h5'


with tf.device('/CPU:0'):
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
        score_list.append((file_name, mean))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print()

    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

    # save to file
    save_path = 'score/' + args.sf
    fo = open(save_path, 'a')
    fo.write('===== begin ====')
    for i, (name, score) in enumerate(score_list):
        name_score = str(i + 1) + ',' + name + " Score = " + str(score)
        fo.write(name_score)
        fo.write('\n')
    fo.write('===== end ====')
    fo.close()



