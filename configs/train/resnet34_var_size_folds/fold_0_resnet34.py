import cv2
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


import matplotlib
matplotlib.use('Agg')

from albumentations import Compose, RandomCrop, ShiftScaleRotate, GaussNoise, ElasticTransform
from albumentations import RandomBrightnessContrast, NoOp
from albumentations.pytorch import ToTensor

from dataflow.datasets import INPUT_PATH, HPADataset, TransformsProgram
from dataflow.dataloaders import get_var_batchsize_train_val_loaders_by_fold
from models.resnet import HPASeparableResNet34
from loss_functions.focal_loss import FocalLoss

from custom_ignite.metrics.accuracy import Accuracy
from custom_ignite.metrics.precision import Precision
from custom_ignite.metrics.recall import Recall
from ignite.metrics import MetricsLambda

seed = 12
device = "cuda"
debug = False

val_fold_index = 0
n_folds = 3

batch_size = 16

train_transforms = Compose([
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.075, rotate_limit=45, interpolation=cv2.INTER_CUBIC, p=0.3),
    ElasticTransform(p=0.5),
    TransformsProgram(
        transforms=[RandomCrop(64, 64), RandomCrop(128, 128), RandomCrop(256, 256), NoOp()],
        milestones=[8000 * batch_size, 16000 * batch_size, 32000 * batch_size],
    ),
    GaussNoise(p=0.1),
    RandomBrightnessContrast(),
    ToTensor(normalize={"mean": [0.5, 0.5, 0.5, 0.5], "std": [1.0, 1.0, 1.0, 1.0]})
])
train_transform_fn = lambda dp: train_transforms(**dp)


val_transforms = Compose([
    RandomBrightnessContrast(),
    ToTensor(normalize={"mean": [0.5, 0.5, 0.5, 0.5], "std": [1.0, 1.0, 1.0, 1.0]})
])
val_transform_fn = lambda dp: val_transforms(**dp)


num_workers = 8

train_loader, val_loader, train_eval_loader = \
    get_var_batchsize_train_val_loaders_by_fold(INPUT_PATH, train_transform_fn, val_transform_fn,
                                                batch_sizes=[batch_size * 8, batch_size * 4, batch_size * 2, batch_size],
                                                milestones=[8000, 16000, 32000],
                                                val_batch_size=16,
                                                num_workers=num_workers, device=device,
                                                fold_index=val_fold_index, n_folds=n_folds,
                                                random_state=seed)

model = HPASeparableResNet34(num_classes=HPADataset.num_tags)


criterion = FocalLoss(gamma=0.4)
optimizer = Adam(model.parameters(), lr=0.00065, weight_decay=0.0001, amsgrad=True)

# Optional config param
lr_scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 90], gamma=0.88)

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
