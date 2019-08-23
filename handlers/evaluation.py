# -*- coding: utf-8 -*
import scipy.stats as stats
import tensorflow as tf


def spearman_corr(y_true, y_pred):
    """
    Spearman's Rank  Correlation Coefficient, SRCC 斯皮尔曼相关性系数
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.py_function(stats.stats.spearmanr,
                          [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout=tf.float32)


def pearson_correlation(y_true, y_pred):
    """
    Linear Correlation Coefficient, LCC 皮尔森相关系数
    :param y_true:
    :param y_pred:
    :return:
    """
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]
