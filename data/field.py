# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np 
import os
import h5py
import warnings



class RawField(object):
    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocesssing(self, x):
        if self.preprocesssing is not None:
            return self.preprocesssing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        if self.postprocessing is not None:
           batch = self.postprocessing(batch)
        return default_collate(batch)


class ImageDetectionsField(RawField):
    def __init__(self, preprocesssing=None, postprocessing=None, detections_path=None, max_dictions=50):
        self.max_dictions = max_dictions
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        super(ImageDetectionsField, self).__init__(preprocesssing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            # features, boxes, cls_prob
            precomp_data = f['%d_features' %image_id][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' %image_id)
            precomp_data = np.random.rand(10,2048)

        delta = self.max_dictions - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_dictions]
        return precomp_data.astype(np.float32)



