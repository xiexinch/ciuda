import os.path as osp
from pathlib import Path

import numpy as np
from mmcv import scandir
from torch.utils.data import Dataset

from mmgen.datasets.builder import DATASETS
from mmgen.datasets.pipelines import Compose

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class RoundImageDataset(Dataset):
    """General unpaired image folder dataset for image generation.

    It assumes that the training directory of images from domain A is
    '/path/to/data/trainA', and that from domain B is '/path/to/data/trainB',
    respectively. '/path/to/data' can be initialized by args 'dataroot'.
    During test time, the directory is '/path/to/data/testA' and
    '/path/to/data/testB', respectively.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, dataroot, pipeline, test_mode=False):
        super().__init__()
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(dataroot), phase + 'A')
        self.dataroot_b = osp.join(str(dataroot), phase + 'B')
        self.dataroot_c = osp.join(str(dataroot), phase + 'C')
        self.data_infos_a = self.load_annotations(self.dataroot_a)
        self.data_infos_b = self.load_annotations(self.dataroot_b)
        self.data_infos_c = self.load_annotations(self.dataroot_c)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)
        self.len_c = len(self.data_infos_c)
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

    def load_annotations(self, dataroot):
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        idx_b = np.random.randint(0, self.len_b)
        img_b_path = self.data_infos_b[idx_b]['path']
        idx_c = np.random.randint(0, self.len_c)
        img_c_path = self.data_infos_c[idx_c]['path']
        results = dict(img_a_path=img_a_path,
                       img_b_path=img_b_path,
                       img_c_path=img_c_path)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        img_b_path = self.data_infos_b[idx % self.len_b]['path']
        img_c_path = self.data_infos_c[idx % self.len_c]['path']
        results = dict(img_a_path=img_a_path,
                       img_b_path=img_b_path,
                       img_c_path=img_c_path)
        return self.pipeline(results)

    def __len__(self):
        return max(self.len_a, self.len_b, self.len_c)

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = scandir(path, suffix=IMG_EXTENSIONS, recursive=True)
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if not self.test_mode:
            return self.prepare_train_data(idx)

        return self.prepare_test_data(idx)
