
import matplotlib
matplotlib.use('Agg')
from albumentations import Compose, CenterCrop, RandomRotate90, Flip
from albumentations.pytorch import ToTensor
from dataflow.datasets import INPUT_PATH, HPADataset
from dataflow.dataloaders import get_test_loader
from models.resnet import HPAResNet50


seed = 12
device = "cuda"
debug = False
write_submission = True

n_tta = 7

tta_transforms = Compose([
    Flip(),
    RandomRotate90(),
    CenterCrop(320, 320),
    ToTensor()
])
tta_transform_fn = lambda dp: tta_transforms(**{"image": dp[0], "id": dp[1]})


batch_size = 128
num_workers = 8

test_loader = get_test_loader(INPUT_PATH, test_transforms=tta_transform_fn,
                              batch_size=batch_size, num_workers=num_workers, device=device)

model = HPAResNet50(num_classes=HPADataset.num_tags)

run_uuid = "6bf2701872df4bd190a9c517a5e52f32"
weights_filename = "model_HPAResNet50_162_val_loss=0.07056979.pth"


