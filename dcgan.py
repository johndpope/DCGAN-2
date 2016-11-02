#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import sys
import os
import math
from dnnlib import utility as ut
from dnnlib import layer as ly
from dnnlib import simple_layer as sly
from dnnlib.config_template import *
from dnnlib.dnn_template import dnn_template
from datetime import datetime

class dcgan(dnn_template):
    def __init__(self,
                 config = {'TrainingConfig' : {'TrainOps' : '',
                                               'LearningRate' : '',
                                               'Type' : 'classified'
                                                },
                           'BatchConfig' : {'TrainNum' : 1000,
                                            'BatchSize' : 50,
                                            'LogPeriod' : 10,
                                            'DataSize' : None},
                           'Data' : None,
                           'StoreConfig' : {'CheckPoint' : './Model/dnn_tempalte.ckpt',
                                            'Initialize' : True}},
                 feature_match = 0.0):
        super(dcgan, self).__init__(config)
        self.feature_match = feature_match

    # あとで消す
    def construct(self):
        # セッションの定義
        self.sess = tf.InteractiveSession()
        # 入出力の定義
        self.io_def()
        # ネットワークの構成
        self.interface()
        # 誤差関数の定義
        self.loss()
        # 学習
        self.training()

        # 精度の定義
        self.get_accuracy()
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()

    # 学習の実行
    def learning(self, data,
                 boost = 5,
                 config = {'BatchConfig' : {'TrainNum' : 100,
                                            'BatchSize' : 50,
                                            'LogPeriod' : 5}}):
        for i in range(config['BatchConfig']['TrainNum']):
            batch = data.train.next_batch(config['BatchConfig']['BatchSize'])
            # 途中経過のチェック
            if i%config['BatchConfig']['LogPeriod'] == 0:
                feed_dict = self.make_feed_dict(prob = True, batch = batch)
                ac0 = self.accuracy[0].eval(feed_dict=feed_dict)
                feed_dict = self.make_feed_dict(prob = True, batch = batch, image = True)
                ac1 = self.accuracy[1].eval(feed_dict=feed_dict)
                print "step %d, D-Loss / G-Loss %g , %g"%(i, ac0, ac1), datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                self.save_checkpoint()
            # 学習
            feed_dict = self.make_feed_dict(prob = False, batch = batch)
            self.d_opt.run(feed_dict=feed_dict)
            for i in range(boost):
                feed_dict = self.make_feed_dict(prob = False, batch = batch, image = True)
                self.g_opt.run(feed_dict=feed_dict)

        self.save_checkpoint()


    # 入出力ベクトルの配置
    def make_feed_dict(self, prob, batch, image = True, z = True):
        feed_dict = {}
        if image:
            feed_dict.setdefault(self.image, batch[0])
        if z:
            feed_dict.setdefault(self.z, batch[1])
        i = 0
        for keep_prob in self.keep_probs:
            if prob:
                feed_dict.setdefault(keep_prob['var'], 1.0)
            else:
                feed_dict.setdefault(keep_prob['var'], keep_prob['prob'])
            i += 1
        return feed_dict


    # 精度評価
    def get_accuracy(self):
        self.accuracy = [self.d_loss, self.g_loss]


    def training(self):
        d_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        g_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        self.d_opt = ut.select_algo(loss_function = self.d_loss,
                                    algo = self.config["TrainingConfig"]["TrainOps"],
                                    learning_rate = self.config["TrainingConfig"]["LearningRate"],
                                    b1 = self.config["TrainingConfig"]["LearningBeta1"],
                                    b2 = self.config["TrainingConfig"]["LearningBeta2"],
                                    var_list = d_val)
        self.g_opt = ut.select_algo(loss_function = self.g_loss,
                                    algo = self.config["TrainingConfig"]["TrainOps"],
                                    learning_rate = self.config["TrainingConfig"]["LearningRate"],
                                    b1 = self.config["TrainingConfig"]["LearningBeta1"],
                                    b2 = self.config["TrainingConfig"]["LearningBeta2"],
                                    var_list = g_val)

    # 誤差関数の定義
    def loss(self):
        config = {'Type' : 'classified-sparse-softmax',
                  'Sparse' : {'Activate' : False,
                              'Logit' : None,
                              'Beta' : None},
                  'BATCH_SIZE' : self.config['BatchConfig']['BatchSize']}

        self.d_loss_real = ut.error_func(y = self.D_REAL,
                                         y_ = tf.constant([1, 0], shape=[self.config['BatchConfig']['BatchSize']], dtype = tf.int64),
                                         config = config)

        self.d_loss_fake = ut.error_func(y = self.D_FAKE,
                                         y_ = tf.constant([0, 1], shape=[self.config['BatchConfig']['BatchSize']], dtype = tf.int64),
                                         config = config)

        self.g_loss_base = ut.error_func(y = self.D_FAKE,
                                    y_ = tf.constant([1, 0], shape=[self.config['BatchConfig']['BatchSize']], dtype = tf.int64),
                                    config = config)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # feature matching
        if self.feature_match != 0.0:
            l = (self.h * self.w * self.c)
            self.g_loss_image = tf.reduce_mean(tf.mul(tf.nn.l2_loss(self.G - self.image) / (l * l) , self.feature_match))
            self.g_loss = self.g_loss_base + self.g_loss_image
        else:
            self.g_loss = self.g_loss_base

    # I/Oの定義
    def io_def(self, h = 64, w = 64, c = 3, zdim = 100):
        self.h = h
        self.w = w
        self.c = c
        self.zdim = zdim
        self.image = tf.placeholder("float", shape=[None, h, w, c])
        self.z = tf.placeholder("float", shape=[None, zdim])
        self.keep_probs = []

    def interface(self):
        self.G = self.generator(z = self.z)
        self.D_REAL = self.discriminator(image = self.image, reuse = False)
        self.D_FAKE = self.discriminator(image = self.G, reuse = True)

    def discriminator(self, image, reuse):
        print "Discriminator: reuse", reuse
        with tf.variable_scope('D', reuse = reuse):
            d1, _ = sly.conv(x = image,
                             vname = 'Conv1',
                             Act = 'LRelu',
                             MaxoutNum = 3,
                             Batch = True,
                             InputNode = [self.h, self.w, self.c],
                             Filter = [5, 5, self.c, 64],
                             Strides = [1, 1, 1, 1],
                             Padding = 'SAME')
            d2 = sly.pooling(x = d1)
            d3, _ = sly.conv(x = d2,
                             vname = 'Conv2',
                             Act = 'LRelu',
                             MaxoutNum = 3,
                             Batch = True,
                             InputNode = [32, 32, 64],
                             Filter = [5, 5, 64, 128],
                             Strides = [1, 1, 1, 1],
                             Padding = 'SAME')
            d4 = sly.pooling(x = d3)
            d5, _ = sly.conv(x = d4,
                             vname = 'Conv3',
                             Act = 'LRelu',
                             MaxoutNum = 3,
                             Batch = True,
                             InputNode = [16, 16, 128],
                             Filter = [5, 5, 128, 256],
                             Strides = [1, 1, 1, 1],
                             Padding = 'SAME')
            d6 = sly.pooling(x = d5)
            d7, _ = sly.conv(x = d6,
                             vname = 'Conv6',
                             Act = 'LRelu',
                             MaxoutNum = 3,
                             Batch = True,
                             InputNode = [8, 8, 256],
                             Filter = [5, 5, 256, 512],
                             Strides = [1, 1, 1, 1],
                             Padding = 'SAME')
            d8 = sly.pooling(x = d7)
            d9 = sly.reshape_tf(x = d8,
                             shape = [-1, 4 * 4 * 512])
            d10, _ = sly.fnn(x = d9, vname = 'FNN',
                             Act = 'Equal',
                             Batch = False,
                             MaxoutNum = 3,
                             InputNode = [4 * 4 * 512],
                             OutputNode = [2])
        return d10

    def generator(self, z):
        print 'Generator'
        with tf.variable_scope('G'):
            g0, _ = sly.project(x = z, vname = 'Project',
                             Act = 'Relu',
                             Batch = True,
                             InputNode = [self.zdim],
                             OutputNode = [4, 4, 1024])
            g1, _ = sly.deconv(x = g0, vname = 'Deconv1',
                            Act = 'Relu',
                            Batch = True,
                            InputNode = [4, 4, 1024],
                            OutputNode = [8, 8, 512],
                            Filter = [5, 5, 512, 1024],
                            Strides = [1, 2, 2, 1],
                            Padding = 'SAME',
                            Network_type = 'transpose')
            g2, _ = sly.deconv(x = g1, vname = 'Deconv2',
                            Act = 'Relu',
                            Batch = True,
                            InputNode = [8, 8, 512],
                            OutputNode = [16, 16, 256],
                            Filter = [5, 5, 256, 512],
                            Strides = [1, 2, 2, 1],
                            Padding = 'SAME',
                            Network_type = 'transpose')
            g3, _ = sly.deconv(x = g2, vname = 'Deconv3',
                               Act = 'Relu',
                               Batch = True,
                               InputNode = [16, 16, 256],
                               OutputNode = [32, 32, 128],
                               Filter = [5, 5, 128, 256],
                               Strides = [1, 2, 2, 1],
                               Padding = 'SAME',
                               Network_type = 'transpose')
            g4, _ = sly.deconv(x = g3, vname = 'Deconv4',
                               Act = 'Tanh',
                               Batch = True,
                               InputNode = [32, 32, 128],
                               OutputNode = [self.h, self.w, self.c],
                               Filter = [5, 5, self.c, 128],
                               Strides = [1, 2, 2, 1],
                               Padding = 'SAME',
                               Network_type = 'transpose')

        return g4

    def get_image(self, z):
        feed_dict = self.make_feed_dict(prob = False, batch = [None, z], image = False)
        result = self.sess.run(self.G, feed_dict = feed_dict)
        return result

if __name__ == '__main__':
    import data_reader
    import cv2
    data = data_reader.read_data_sets()
    config = dnn_cell_template(data = None, length = None)
    config["TrainingConfig"]["LearningRate"] = 0.0002
    config["TrainingConfig"]["LearningBeta1"] = 0.5
    dnn = dcgan(config = config, feature_match = 0.1)
    dnn.construct()
    learning_config = {'BatchConfig' : {'TrainNum' : 3,
                                        'BatchSize' : 50,
                                        'LogPeriod' : 1}}
    dnn.learning(data = data, config = learning_config, boost = 1)
    z = [2.0 * np.random.rand(100) - 1.0]
    img = dnn.get_image(z = z)
    cv2.imwrite('sample.png', img[0])
    dnn.session_close()
