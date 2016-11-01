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
                                            'Initialize' : True}}):
        super(dcgan, self).__init__(config)

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
        '''
        # 精度の定義
        self.get_accuracy()
        # チェックポイントの呼び出し
        self.saver = tf.train.Saver()
        self.restore()
        '''

    def training(self):
        d_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        g_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        self.d_opt = ut.select_algo(loss_function = self.d_loss,
                                    algo = self.config["TrainingConfig"]["TrainOps"],
                                    learning_rate = self.config["TrainingConfig"]["LearningRate"],
                                    var_list = d_val)
        self.g_opt = ut.select_algo(loss_function = self.g_loss,
                                    algo = self.config["TrainingConfig"]["TrainOps"],
                                    learning_rate = self.config["TrainingConfig"]["LearningRate"],
                                    var_list = g_val)
        

    # 誤差関数の定義
    def loss(self):
        config = {'Type' : 'classified-sigmoid',
                  'Sparse' : {'Activate' : False,
                              'Logit' : None,
                              'Beta' : None},
                  'BATCH_SIZE' : self.config['BatchConfig']['BatchSize']}
        self.d_loss_real = ut.error_func(y = self.D_REAL,
                                         y_ = tf.ones_like(self.D_REAL),
                                         config = config)

        self.d_loss_fake = ut.error_func(y = self.D_FAKE,
                                         y_ = tf.zeros_like(self.D_FAKE),
                                         config = config)

        self.g_loss = ut.error_func(y = self.D_FAKE,
                                    y_ = tf.ones_like(self.D_FAKE),
                                    config = config)
        self.d_loss = self.d_loss_real + self.d_loss_fake


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
                             OutputNode = [1])
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
                               Act = 'Relu',
                               Batch = True,
                               InputNode = [32, 32, 128],
                               OutputNode = [self.h, self.w, self.c],
                               Filter = [5, 5, self.c, 128],
                               Strides = [1, 2, 2, 1],
                               Padding = 'SAME',
                               Network_type = 'transpose')

        return g4



if __name__ == '__main__':
    import data_reader
    data = data_reader.read_data_sets()
    config = dnn_cell_template(data = None, length = None)
    dnn = dcgan(config = config)
    dnn.construct()
    dnn.session_close()
