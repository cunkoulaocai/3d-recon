import itertools
import json
import os

import cv2
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

from data_loaders import binvox_rw

dir_path = os.path.dirname(os.path.realpath(__file__))

SHAPENET_VOX_PATH = os.path.join(dir_path, "../..", "ShapeNetVox32")


class ZSampler(object):
    def __init__(self, use_normal=False):
        self.use_normal = use_normal

    def __call__(self, batch_size, z_dim):
        if self.use_normal:
            ret = np.random.normal(size=[batch_size, z_dim])
        else:
            ret = np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
        return ret.astype(np.float32)


class PSampler(object):
    def __call__(self, batch_size):
        p12 = np.random.normal(size=[batch_size, 2])
        p3 = np.random.uniform(-1.0, 1.0, [batch_size, 1])
        ret = np.concatenate([p12, p3], axis=-1)
        return ret.astype(np.float32)


class ValDataset(data.Dataset):

    def __init__(self, data_settings, shape=(32, 32, 1), viewpoints=10, binarize_data=True):
        self.shape = shape[:3]
        self.viewpoints = viewpoints
        self.binarize_data = binarize_data

        self.pose_info = {}
        self.imgs = []

        for data_dir in data_settings:
            pose_info = json.load(open(os.path.join(data_dir, "pose_info.json")))
            self.pose_info.update(pose_info)
            self.imgs += [(data_dir, fname) for fname in pose_info.keys()]

    def _fname_to_pose_(self, fname):
        return np.array([
            self.pose_info[fname]['rx'],
            self.pose_info[fname]['ry'],
            self.pose_info[fname]['rz']
        ]).astype(np.float32)

    def _fname_to_vox_(self, fname):
        """fname in pose_info.keys()"""
        sid, mid = fname.split("_")[:2]
        # k = "%s_%s"%(sid, mid)
        ret = None
        with open(os.path.join(SHAPENET_VOX_PATH, sid, mid, "model.binvox"), "rb") as f:
            md = binvox_rw.read_as_3d_array(f)
            ret = md.data.astype(np.float32)
        return ret

    def _fname_to_img_(self, data_dir, fname):
        img = cv2.imread(os.path.join(data_dir, fname), 0)[..., np.newaxis].astype(np.float32)
        if self.binarize_data:
            img = (img > 0).astype(np.float32) * 255
        img /= 255.
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data_dir, fname = self.imgs[index]
        vox = self._fname_to_vox_(fname).astype(np.float32)
        pose = self._fname_to_pose_(fname).astype(np.float32)
        img = self._fname_to_img_(data_dir, fname).astype(np.float32)
        return {
            "image": img,
            "vox": vox,
            "pose": pose
        }


class TrainDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_setting, shape=[32, 32, 1], viewpoints=10, binarize_data=False,
                 disjoint_split=False):
        self.shape = shape[:3]
        self.viewpoints = viewpoints
        self.binarize_data = binarize_data

        self.pose_info = {}  # abs path -> pose information
        self.modelid_withpose = set()  # all model id for pos supervision
        self.modelid_nopose = set()  # all model id for non-pose supervision
        self.models = {}  # mode_id -> list of images in absolute path

        for data_dir, p_ratio, nop_ratio in data_setting:
            # Update pose information
            pose_info = json.load(open(os.path.join(data_dir, "pose_info.json")))
            self.pose_info.update({
                os.path.join(data_dir, k): pose_info[k] for k in pose_info.keys()
            })

            # get model keys from this data_dir
            model_keys = set()
            for fname in pose_info.keys():
                sid, mid = fname.split("_")[:2]
                model_id = "%s-%s" % (sid, mid)
                model_keys.add(model_id)
                if model_id not in self.models:
                    self.models[model_id] = []
                if len(self.models[model_id]) < self.viewpoints:
                    self.models[model_id].append(os.path.join(data_dir, fname))
            model_keys = list(model_keys)

            # splits
            n = int(len(model_keys) * p_ratio)
            m = int(len(model_keys) * nop_ratio)

            for mid in model_keys[:n]:
                self.modelid_withpose.add(mid)

            if disjoint_split:
                assert n + m <= len(model_keys)
                for mid in model_keys[n:n + m]:
                    self.modelid_nopose.add(mid)
            else:
                for mid in model_keys[:m]:
                    self.modelid_nopose.add(mid)

        self.vp_pairs = itertools.product(range(self.viewpoints), range(self.viewpoints))
        self.vp_pairs = [(i, j) for i, j in self.vp_pairs if not (i > j)]

        self.modelid_withpose = list(self.modelid_withpose)
        self.modelid_nopose = list(self.modelid_nopose)

        print("Dataset with pose:%d" % len(self.modelid_withpose))
        print("Dataset no pose  :%d" % len(self.modelid_nopose))

        self.uris_super = []
        self.uris_unsuper = []

        for mid_pose in self.modelid_withpose:
            for i, j in self.vp_pairs:
                fname_i, fname_j = self.models[mid_pose][i], self.models[mid_pose][j]
                self.uris_super.append([fname_i, fname_j])

        for mid_nop in self.modelid_nopose:
            for i in range(self.viewpoints):
                fname_u = self.models[mid_nop][i]
                self.uris_unsuper.append(fname_u)

    def __getitem__(self, index):
        'Generates one sample of data'
        id_super = np.random.randint(0, len(self.uris_super))
        id_unsuper = np.random.randint(0, len(self.uris_unsuper))

        fname_i, fname_j = self.uris_super[id_super]
        fname_u = self.uris_unsuper[id_unsuper]

        img = self._fname_to_img_(fname_u)
        img_i, img_j = self._fname_to_img_(fname_i), self._fname_to_img_(fname_j)
        pos_i, pos_j = self._fname_to_pose_(fname_i), self._fname_to_pose_(fname_j)
        return {
            "img_1": img_i,
            "img_2": img_j,
            "pos_1": pos_i,
            "pos_2": pos_j,
            'img': img
        }

    def _fname_to_img_(self, fname):
        img = cv2.imread(fname, 0)[..., np.newaxis].astype(np.float32)
        if self.binarize_data:
            img = (img > 0).astype(np.float32) * 255
        img /= 255.
        return img

    def _fname_to_pose_(self, fname):
        return np.array([
            self.pose_info[fname]['rx'],
            self.pose_info[fname]['ry'],
            self.pose_info[fname]['rz']
        ]).astype(np.float32)

    def __len__(self):
        return max(len(self.uris_super), len(self.uris_unsuper))


class TrainDataLoader(DataLoader):
    def __init__(self, data_setting, shape, viewpoints, binarize_data, batch_size=32, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=8):
        dataset = TrainDataset(data_setting, shape, viewpoints, binarize_data)
        super(TrainDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers)

        self.enumerater = enumerate(self)

    def next_batch(self):
        try:
            _, batch = self.enumerater.next()
        except:
            self.enumerater = enumerate(self)
            _, batch = self.enumerater.next()

        return batch


class ValDataLoader(DataLoader):
    def __init__(self, data_setting, shape, viewpoints, binarize_data, batch_size=32, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=8):
        dataset = ValDataset(data_setting, shape, viewpoints, binarize_data)
        super(ValDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers)

        self.enumerater = enumerate(self)

    def next_batch(self):
        try:
            _, batch = self.enumerater.next()
        except:
            self.enumerater = enumerate(self)
            _, batch = self.enumerater.next()

        return batch
