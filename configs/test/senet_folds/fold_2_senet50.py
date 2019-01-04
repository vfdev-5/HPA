
import matplotlib
matplotlib.use('Agg')
from albumentations import Compose, ElasticTransform, RandomRotate90, Flip
from albumentations.pytorch import ToTensor
from dataflow.datasets import INPUT_PATH, HPADataset
from dataflow.dataloaders import get_test_loader
from models.senet import HPASENet50

seed = 12
device = "cuda"
debug = False
write_submission = True

n_tta = 7

tta_transforms = Compose([
    Flip(),
    RandomRotate90(),
    ElasticTransform(p=0.3),
    ToTensor()
])
tta_transform_fn = lambda dp: tta_transforms(**dp)

batch_size = 96
num_workers = 8

test_loader = get_test_loader(INPUT_PATH, test_transforms=tta_transform_fn,
                              batch_size=batch_size, num_workers=num_workers, device=device)

model = HPASENet50(num_classes=HPADataset.num_tags)

run_uuid = "5ce53937244e43079259f35872b1ebfa"
weights_filename = "model_HPASENet50_49_val_loss=0.07271714.pth"
