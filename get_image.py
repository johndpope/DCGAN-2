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
from dcgan import dcgan



import data_reader
import cv2
data = data_reader.read_data_sets()
config = dnn_cell_template(data = None, length = None)
config["TrainingConfig"]["LearningRate"] = 0.0002
config["TrainingConfig"]["LearningBeta1"] = 0.5
config["StoreConfig"]["Initialize"] = False
dnn = dcgan(config = config, feature_match = 0.1)
dnn.construct()
learning_config = {'BatchConfig' : {'TrainNum' : 100000,
                                    'BatchSize' : 50,
                                    'LogPeriod' : 10}}
z = []
for i in range(100):
    z.append(2.0 * np.random.rand(100) - 1.0)
img = dnn.get_image(z = np.array(z))
print img
img = np.array(img)
for i in range(100):
    p = img[i]
    cv2.imwrite('./Pic/sample' + str(i) +'.png', p)
dnn.session_close()
