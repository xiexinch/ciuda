from mmseg.datasets import DATASETS
from mmseg.datasets.builder import build_dataset

from mmgen.datasets.builder import build_dataset as build_mmgen_dataset
from mmgen.datasets.pipelines import Compose


@DATASETS.register_module()
class MixDataset(object):

    def __init__(self, city: dict, dn_pair_list: str, zurich_img_dir: str,
                 zurich_pipeline: dict):

        self.city = build_dataset(city)
        self.zurich_img_dir = zurich_img_dir
        self.dn_pair_list = dn_pair_list
        self.dn_pair = []

        self.zurich_pipeline = Compose(zurich_pipeline)

        self.ignore_index = self.city.ignore_index
        self.CLASSES = self.city.CLASSES
        self.PALETTE = self.city.PALETTE

    def load_annotations(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []

        with open(self.dn_pair_list, 'r') as f:
            line = f.readline()
            while line is not None:
                dn = line.split(',')
                n = self.zurich_img_dir + dn[0] + self.target_img_suffix
                d = self.zurich_img_dir + dn[1] + self.target_img_suffix
                line = f.readline()
                self.dn_pair.append((d, n))

        pair_paths = sorted(self.scan_folder(self.dataroot))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = ''

    def __getitem__(self, idx):
        city = self.city[idx]
        d, n = self.dn_pair[idx % len(self.dn_pair)]
        day = dict(img_info=dict(filename=d))
        night = dict(img_info=dict(filename=n))
        self.pre_pipeline(day)
        self.pre_pipeline(night)
        day_img = self.zurich_pipeline(day)
        night_img = self.zurich_pipeline(night)

        return {
            **city,
            'day_img_metas': day_img['img_metas'],
            'day_img': day_img['img'],
            'night_img_metas': night_img['img_metas'],
            'night_img': night_img['img'],
        }

    def __len__(self):
        return len(self.city)
