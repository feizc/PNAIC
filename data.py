# data processing methods, partially copied from https://github.com/aimagelab/meshed-memory-transformer

from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import shutil
import warnings
import h5py


class RawField(object):
    """ A general datatype"""

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def prepocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None,
                 max_detections=100, sort_by_prob=False, load_in_tmp=True):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        tmp_detections_path = os.path.join('\tmp', os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage('/tmp')[-1] < os.path.getsize(detections_path):
                    warnings.warn('Loading from %s, /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
                    
        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            precomp_data = f['%d_features' % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[np.argsort(np.max( f['%d_features' % image_id][()], -1))[::-1]]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:, self.max_detections]

        return precomp_data.astype(np.float32)


class Vocab(object):
    """Create a Vocab object from collections.Counter"""

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None):



class TextField(RawField):
    a=10























