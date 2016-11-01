#! /usr/bin/env python
# -*- coding:utf-8 -*-
import commands
import csv
import numpy as np
import cv2

class DataSet(object):
    def __init__(self, zdim = 100):
        self.zdim = zdim
        cmd = "find ./Data/Sample -name \'*.jpg\'"
        res = commands.getoutput(cmd)
        self.files = res.split("\n")
        self.start = 0
        imgs, labels= [], []
        for i in range(len(self.files) - 1):
            img = self.files[i]
            imgs.append(i)
            labels.append(i)
        self._images = np.array(imgs)
        self._labels = np.array(labels)


    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels


    def next_batch(self, batch_size):
        start = self.start
        if self.start + batch_size >= len(self._images):
            print "Next epoch"
            perm = np.arange(len(self._images))
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            end = batch_size
        else:
            end = min(self.start + batch_size, len(self._images) - 1)
        self.start = end
        imgs, labels = [], []
        for i in range(start, end):
            img = cv2.imread(self.files[i])
            img = cv2.resize(img, (64, 64))
            imgs.append(img)
            r = 2.0 * np.random.rand(self.zdim) - 1.0
            labels.append(r)

        return [np.array(imgs), np.array(labels)]

def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet()

    return data_sets

if __name__ == '__main__':
    data = read_data_sets()
    for i in range(100):
        fs = data.train.next_batch(50)
        print fs[0][0].shape, fs[1][0].shape
