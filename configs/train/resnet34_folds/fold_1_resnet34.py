import cv2
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


import matplotlib
matplotlib.use('Agg')

from albumentations import Compose, RandomCrop, RandomCropNearBBox, ShiftScaleRotate, GaussNoise, ElasticTransform
from albumentations import CenterCrop
from albumentations.pytorch import ToTensor

from dataflow.datasets import INPUT_PATH, HPADataset
from dataflow.dataloaders import get_base_train_val_loaders_by_fold
from models.senet import HPASENet50
from loss_functions.focal_loss import FocalLoss

from custom_ignite.metrics.accuracy import Accuracy
from custom_ignite.metrics.precision import Precision
from custom_ignite.metrics.recall import Recall
from ignite.metrics import MetricsLambda

seed = 12
device = "cuda"
debug = False

val_fold_index = 1
n_folds = 3

train_transforms = Compose([
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.01, rotate_limit=15, interpolation=cv2.INTER_CUBIC, p=0.3),
    ElasticTransform(p=0.5),
    ToTensor()
])
train_transform_fn = lambda dp: train_transforms(**dp)


val_transforms = Compose([
    ToTensor()
])
val_transform_fn = lambda dp: val_transforms(**dp)


batch_size = 8
num_workers = 8

train_loader, val_loader, train_eval_loader = \
    get_base_train_val_loaders_by_fold(INPUT_PATH, train_transform_fn, val_transform_fn,
                                       batch_size=batch_size, num_workers=num_workers, device=device,
                                       fold_index=val_fold_index, n_folds=n_folds,
                                       random_state=seed)

model = HPASENet50(num_classes=HPADataset.num_tags)


criterion = FocalLoss(gamma=0.4)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)

# Optional config param
lr_scheduler = MultiStepLR(optimizer, milestones=[20, 30, 40, 50, 60, 70], gamma=0.77)

num_epochs = 100


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(torch.sigmoid(y_pred))
    return y_pred, y


# Optional config param
accuracy_metric = Accuracy(output_transform=thresholded_output_transform,
                           is_multilabel=True)
precision_metric = Precision(output_transform=thresholded_output_transform,
                             average=False, is_multilabel=True)
recall_metric = Recall(output_transform=thresholded_output_transform,
                       average=False, is_multilabel=True)
f1_metric = precision_metric * recall_metric * 2 / (recall_metric + precision_metric + 1e-20)

precision_metric = MetricsLambda(lambda t: torch.mean(t).item(), precision_metric)
recall_metric = MetricsLambda(lambda t: torch.mean(t).item(), recall_metric)
f1_metric = MetricsLambda(lambda t: torch.mean(t).item(), f1_metric)

metrics = {
    "precision": precision_metric,
    "recall": recall_metric,
    "accuracy": accuracy_metric,
    "f1": f1_metric
}

log_interval = 50
val_interval_epochs = 2
val_metrics = metrics

trainer_checkpoint_interval = 5000
