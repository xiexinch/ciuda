from torch.utils.data import Dataset
# from mmseg.datasets import DATASETS
from mmseg.datasets.builder import build_dataset
from mmgen.datasets.pipelines import Compose
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer


@DATASETS.register_module()
class MixDataset2(Dataset):

    def __init__(self, city: dict, zurich: dict, pipeline: dict):

        self.city = build_dataset(city)
        self.zurich = build_dataset(zurich)
        self.pipeline = Compose(pipeline)

        self.ignore_index = self.city.ignore_index
        self.CLASSES = self.city.CLASSES
        self.PALETTE = self.city.PALETTE

    def __getitem__(self, idx):
        city = self.city[idx % len(self)]
        zurich = self.zurich[idx]
        _data = {**city, **zurich}

        return self.pipeline(_data)

    def __len__(self):
        return len(self.zurich)
