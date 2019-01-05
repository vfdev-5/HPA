import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset

from dataflow.datasets import HPADataset, TransformedDataset


def get_train_val_indices(trainval_df, fold_index=0, n_splits=3, random_state=None, return_targets=False):

    image_ids = list(trainval_df['Id'])
    y = np.zeros((len(image_ids), HPADataset.num_tags), dtype=np.uint8)
    for i, tags in enumerate(trainval_df['Target']):
        tags = tags.split(" ")
        for t in tags:
            y[i, int(t)] = 1

    skf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state)
    x = y
    train_fold_indices = None
    val_fold_indices = None
    for i, (train_indices, val_indices) in enumerate(skf.split(x, y)):
        if i == fold_index:
            train_fold_indices = train_indices
            val_fold_indices = val_indices
            break
    assert val_fold_indices is not None and train_fold_indices is not None
    if return_targets:
        return train_fold_indices, val_fold_indices, y
    return train_fold_indices, val_fold_indices


def get_base_train_val_loaders_by_fold(input_path, train_transforms, val_transforms,
                                       batch_size=16, num_workers=8, device="cuda",
                                       val_batch_size=None,
                                       fold_index=0, n_folds=3, random_state=None,
                                       limit_train_num_samples=None,
                                       limit_val_num_samples=None):
    trainval_df = pd.read_csv(input_path / "train.csv")
    trainval_ds = HPADataset(trainval_df, input_path / "train")
    train_fold_indices, val_fold_indices = get_train_val_indices(trainval_df,
                                                                 fold_index=fold_index,
                                                                 n_splits=n_folds,
                                                                 random_state=random_state)

    if limit_train_num_samples is not None:
        train_fold_indices = train_fold_indices[:limit_train_num_samples]

    if limit_val_num_samples is not None:
        val_fold_indices = val_fold_indices[:limit_val_num_samples]

    train_ds = Subset(trainval_ds, train_fold_indices)
    val_ds = Subset(trainval_ds, val_fold_indices)
    train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    train_loader = DataLoader(train_ds, shuffle=True,
                              batch_size=batch_size, num_workers=num_workers,
                              pin_memory="cuda" in device, drop_last=True)

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = DataLoader(val_ds, shuffle=False,
                            batch_size=val_batch_size, num_workers=num_workers,
                            pin_memory="cuda" in device, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False,
                                   batch_size=val_batch_size, num_workers=num_workers,
                                   pin_memory="cuda" in device, drop_last=False)

    return train_loader, val_loader, train_eval_loader


def get_resampled_train_val_loaders_by_fold(input_path, train_transforms, val_transforms,
                                            batch_size=16, num_workers=8, device="cuda",
                                            fold_index=0, n_folds=3, random_state=None,
                                            limit_train_num_samples=None,
                                            limit_val_num_samples=None):

    trainval_df = pd.read_csv(input_path / "train.csv")
    trainval_ds = HPADataset(trainval_df, input_path / "train")
    train_fold_indices, val_fold_indices, y = get_train_val_indices(trainval_df,
                                                                    fold_index=fold_index,
                                                                    n_splits=n_folds,
                                                                    random_state=random_state,
                                                                    return_targets=True)
    tag_weights = np.power(y.sum(axis=0), 0.77)
    tag_weights = 20.0 / tag_weights

    if limit_train_num_samples is not None:
        train_fold_indices = train_fold_indices[:limit_train_num_samples]

    if limit_val_num_samples is not None:
        val_fold_indices = val_fold_indices[:limit_val_num_samples]

    train_ds = Subset(trainval_ds, train_fold_indices)
    val_ds = Subset(trainval_ds, val_fold_indices)
    train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    sample_weights = np.dot(y[train_fold_indices, :], tag_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, sampler=sampler,
                              batch_size=batch_size, num_workers=num_workers,
                              pin_memory="cuda" in device, drop_last=True)

    val_loader = DataLoader(val_ds, shuffle=False,
                            batch_size=batch_size, num_workers=num_workers,
                            pin_memory="cuda" in device, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False,
                                   batch_size=batch_size, num_workers=num_workers,
                                   pin_memory="cuda" in device, drop_last=False)

    return train_loader, val_loader, train_eval_loader


def get_test_loader(input_path, test_transforms,
                    batch_size=16, num_workers=8, device="cuda"):

    test_df = pd.read_csv(input_path / "sample_submission.csv")
    test_ds = HPADataset(test_df, input_path / "test")
    test_ds = TransformedDataset(test_ds, transform_fn=test_transforms)
    test_loader = DataLoader(test_ds, shuffle=False,
                             batch_size=batch_size, num_workers=num_workers,
                             pin_memory="cuda" in device, drop_last=False)
    return test_loader


def get_ae_loader(input_path, transforms=None,
                  batch_size=16, num_workers=8, device="cuda"):

    trainval_df = pd.read_csv(input_path / "train.csv")
    trainval_ds = HPADataset(trainval_df, input_path / "train")
    test_df = pd.read_csv(input_path / "sample_submission.csv")
    test_ds = HPADataset(test_df, input_path / "test")

    combined_ds = ConcatDataset([trainval_ds, test_ds])

    if transforms is not None:
        combined_ds = TransformedDataset(combined_ds, transform_fn=transforms)

    return DataLoader(combined_ds, shuffle=True,
                      batch_size=batch_size, num_workers=num_workers,
                      pin_memory="cuda" in device, drop_last=True)


class VariableSizeBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_sizes, milestones, drop_last):
        assert len(batch_sizes) == len(milestones) + 1
        assert sorted(milestones) == milestones
        super(VariableSizeBatchSampler, self).__init__(sampler, batch_sizes[0], drop_last)
        self._count = 0
        self.batch_sizes = batch_sizes
        self.milestones = milestones

    def __iter__(self):
        super(VariableSizeBatchSampler).__iter__()
        self._count += 1
        idx = sum([m <= self._count for m in self.milestones])
        self.batch_size = self.batch_sizes[idx]


def get_var_batchsize_train_val_loaders_by_fold(input_path, train_transforms, val_transforms,
                                                batch_sizes=[], milestones=[],
                                                val_batch_size=16,
                                                num_workers=8, device="cuda",
                                                fold_index=0, n_folds=3, random_state=None,
                                                limit_train_num_samples=None,
                                                limit_val_num_samples=None):
    assert len(batch_sizes) == len(milestones) + 1
    assert sorted(milestones) == milestones

    trainval_df = pd.read_csv(input_path / "train.csv")
    trainval_ds = HPADataset(trainval_df, input_path / "train")
    train_fold_indices, val_fold_indices = get_train_val_indices(trainval_df,
                                                                 fold_index=fold_index,
                                                                 n_splits=n_folds,
                                                                 random_state=random_state)

    if limit_train_num_samples is not None:
        train_fold_indices = train_fold_indices[:limit_train_num_samples]

    if limit_val_num_samples is not None:
        val_fold_indices = val_fold_indices[:limit_val_num_samples]

    train_ds = Subset(trainval_ds, train_fold_indices)
    val_ds = Subset(trainval_ds, val_fold_indices)
    train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    sampler = RandomSampler(train_ds)
    batch_sampler = VariableSizeBatchSampler(sampler, batch_sizes, milestones, drop_last=True)

    train_loader = DataLoader(train_ds,
                              batch_sampler=batch_sampler,
                              num_workers=num_workers,
                              pin_memory="cuda" in device)

    val_loader = DataLoader(val_ds, shuffle=False,
                            batch_size=val_batch_size * 4, num_workers=num_workers,
                            pin_memory="cuda" in device, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False,
                                   batch_size=val_batch_size * 4, num_workers=num_workers,
                                   pin_memory="cuda" in device, drop_last=False)

    return train_loader, val_loader, train_eval_loader
