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
from models.resnet import HPAResNet50

from custom_ignite.metrics.accuracy import Accuracy
from custom_ignite.metrics.precision import Precision
from custom_ignite.metrics.recall import Recall

seed = 12
device = "cuda"
debug = False

val_fold_index = 0
n_folds = 3

train_transforms = Compose([
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.01, rotate_limit=15, interpolation=cv2.INTER_CUBIC, p=0.3),
    ElasticTransform(p=0.3),
    RandomCrop(256, 256),
    ToTensor()
])
train_transform_fn = lambda dp: train_transforms(**{"image": dp[0], "tags": dp[1].astype('float32')})


val_transforms = Compose([
    CenterCrop(256, 256),
    ToTensor()
])
val_transform_fn = lambda dp: val_transforms(**{"image": dp[0], "tags": dp[1].astype('float32')})


batch_size = 20
num_workers = 8

train_loader, val_loader, train_eval_loader = \
    get_base_train_val_loaders_by_fold(INPUT_PATH, train_transform_fn, val_transform_fn,
                                       batch_size=batch_size, num_workers=num_workers, device=device,
                                       fold_index=val_fold_index, n_folds=n_folds,
                                       random_state=seed)

model = HPAResNet50(num_classes=HPADataset.num_tags)


# Training
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


criterion = FocalLoss(gamma=0.5)
optimizer = SGD(model.parameters(), lr=0.02, momentum=0.5)

# Optional config param
lr_scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

num_epochs = 100


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(torch.sigmoid(y_pred))
    return y_pred, y


# Optional config param
precision_metric = Precision(output_transform=thresholded_output_transform,
                             average=True, is_multilabel=True)
recall_metric = Recall(output_transform=thresholded_output_transform,
                       average=True, is_multilabel=True)

metrics = {
    "precision": precision_metric,
    "recall": recall_metric,
    "accuracy": Accuracy(output_transform=thresholded_output_transform,
                         is_multilabel=True)
}

log_interval = 10
val_interval_epochs = 1
val_metrics = metrics

trainer_checkpoint_interval = 1000
