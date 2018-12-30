import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from albumentations import Compose, RandomCrop, RandomCropNearBBox, ShiftScaleRotate, GaussNoise, ElasticTransform
from albumentations import CenterCrop, Rotate, RandomRotate90, Flip
from albumentations.pytorch import ToTensor
from dataflow.datasets import INPUT_PATH, HPADataset
from dataflow.dataloaders import get_base_train_val_loaders_by_fold, get_train_val_indices, Subset, TransformedDataset, DataLoader
from models.ae_resnet import HPAAutoEncodeResNet34


seed = 12
device = "cuda"
debug = False

val_fold_index = 0
n_folds = 3

n_tta = 7

tta_transforms = Compose([
    Flip(),
    RandomRotate90(),
    CenterCrop(320, 320),
    ToTensor()
])
tta_transform_fn = lambda dp: tta_transforms(**{"image": dp[0], "tags": dp[1].astype('float32')})


batch_size = 64
num_workers = 8

trainval_df = pd.read_csv(INPUT_PATH / "train.csv")
trainval_ds = HPADataset(trainval_df, INPUT_PATH / "train")
_, val_fold_indices = get_train_val_indices(trainval_df,
                                            fold_index=val_fold_index,
                                            n_splits=n_folds,
                                            random_state=seed)


val_ds = Subset(trainval_ds, val_fold_indices)
val_ds = TransformedDataset(val_ds, transform_fn=tta_transform_fn)


val_loader = DataLoader(val_ds, shuffle=False,
                        batch_size=batch_size, num_workers=num_workers,
                        pin_memory="cuda" in device, drop_last=False)

model = HPAAutoEncodeResNet34(num_classes=HPADataset.num_tags)

run_uuid = "4779d331e2ad496c85e0153e755078b9"
weights_filename = "checkpoint_HPAAutoEncodeResNet34_5000.pth"
