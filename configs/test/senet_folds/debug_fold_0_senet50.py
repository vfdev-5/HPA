import pandas as pd
import matplotlib
matplotlib.use('Agg')
from albumentations import Compose, ElasticTransform, RandomRotate90, Flip
from albumentations.pytorch import ToTensor
from dataflow.datasets import INPUT_PATH, HPADataset, TransformedDataset
from dataflow.dataloaders import get_test_loader, DataLoader, Subset
from models.senet import HPASENet50

seed = 12
device = "cuda"
debug = True
write_submission = True

n_tta = 2

tta_transforms = Compose([
    Flip(),
    RandomRotate90(),
    ElasticTransform(p=0.3),
    ToTensor()
])
tta_transform_fn = lambda dp: tta_transforms(**dp)

batch_size = 96
num_workers = 8

test_df = pd.read_csv(INPUT_PATH / "sample_submission.csv")
test_ds = HPADataset(test_df, INPUT_PATH / "test")

test_ds = Subset(test_ds, list(range(150)))

test_ds = TransformedDataset(test_ds, transform_fn=tta_transform_fn)
test_loader = DataLoader(test_ds, shuffle=False,
                         batch_size=batch_size, num_workers=num_workers,
                         pin_memory="cuda" in device, drop_last=False)

model = HPASENet50(num_classes=HPADataset.num_tags)

run_uuid = "790dea0b21704cb5b7b6c6381f9361d6"
weights_filename = "model_HPASENet50_50_val_loss=0.07531988.pth"
