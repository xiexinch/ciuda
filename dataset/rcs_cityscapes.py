import json
import os.path as osp

import mmcv
import numpy as np
import torch
from mmseg.datasets import CityscapesDataset, DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(overall_class_stats.items(),
                           key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class RCSCityscapesDataset(CityscapesDataset):

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 rcs_cfg=None,
                 **kwargs):
        super().__init__(img_suffix, seg_map_suffix, **kwargs)

        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                self.data_root, self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(osp.join(self.data_root, 'samples_with_class.json'),
                      'r') as f:
                samples_with_class_and_n = json.load(f)

            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file_, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file_.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos):
                file_ = dic['ann']['seg_map'].split('/')[-1]
                self.file_to_idx[file_] = i

    def __getitem__(self, idx):
        if self.rcs_enabled:
            c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
            f1 = np.random.choice(self.samples_with_class[c])
            idx_ = self.file_to_idx[f1]
            if self.test_mode:
                return self.prepare_test_img(idx_)
            else:
                return self.prepare_train_img(idx_)
        else:
            return super().__getitem__(idx)
