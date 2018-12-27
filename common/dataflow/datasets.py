from pathlib import Path

import numpy as np
import pandas as pd

import cv2

from torch.utils.data import Dataset


INPUT_PATH = Path("__file__").parent.parent.parent / "input"


def get_png_image(image_id, path):
    r_img_path = Path(path) / "{}_red.png".format(image_id)
    b_img_path = Path(path) / "{}_blue.png".format(image_id)
    y_img_path = Path(path) / "{}_yellow.png".format(image_id)
    g_img_path = Path(path) / "{}_green.png".format(image_id)

    r_img = cv2.imread(r_img_path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    assert r_img is not None, "Failed to read image: {}".format(r_img_path)
    b_img = cv2.imread(b_img_path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    assert b_img is not None, "Failed to read image: {}".format(b_img_path)
    y_img = cv2.imread(y_img_path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    assert y_img is not None, "Failed to read image: {}".format(y_img_path)
    g_img = cv2.imread(g_img_path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    assert g_img is not None, "Failed to read image: {}".format(g_img_path)
    return np.concatenate([r_img[:, :, None],
                           b_img[:, :, None],
                           g_img[:, :, None],
                           y_img[:, :, None]], axis=-1)


def get_tif_image(image_id, path):
    raise NotImplementedError()


class HPADataset(Dataset):

    tags = ['Actin filaments', 'Aggresome', 'Cell junctions', 'Centrosome',
            'Cytokinetic bridge', 'Cytoplasmic bodies', 'Cytosol', 'Endoplasmic reticulum',
            'Endosomes', 'Focal adhesion sites', 'Golgi apparatus', 'Intermediate filaments',
            'Lipid droplets', 'Lysosomes', 'Microtubule ends', 'Microtubule organizing center',
            'Microtubules', 'Mitochondria', 'Mitotic spindle', 'Nuclear bodies', 'Nuclear membrane',
            'Nuclear speckles', 'Nucleoli', 'Nucleoli fibrillar center', 'Nucleoplasm',
            'Peroxisomes', 'Plasma membrane', 'Rods and rings']
    num_tags = len(tags)

    def __init__(self, dataframe, path, mode='png'):
        assert isinstance(dataframe, pd.DataFrame)
        path = Path(path)
        assert path.exists()
        assert mode in ('png', 'tif')
        self.path = path
        self.image_ids = list(dataframe['Id'])

        if "Target" in dataframe:
            self.targets = np.zeros((len(self.image_ids), self.num_tags), dtype=np.uint8)
            for i, tags in enumerate(dataframe['Target']):
                tags = tags.split(" ")
                for t in tags:
                    self.targets[i, int(t)] = 1
        else:
            self.targets = self.image_ids

        self.get_image_fn = get_png_image if mode == 'png' else get_tif_image
        self.mode = mode

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.get_image_fn(image_id, self.path)
        target = self.targets[index]
        return img, target


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(dp)
