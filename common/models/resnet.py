
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class HPAResNet50(nn.Module):

    def __init__(self, num_classes):
        super(HPAResNet50, self).__init__()

        self.r50 = resnet50(pretrained=False)

        def create_features(r50):
            return nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
                r50.bn1,
                r50.relu,
                r50.maxpool,
                r50.layer1,
                r50.layer2,
                r50.layer3,
                r50.layer4
            )

        self.r50_features = create_features(self.r50)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        size = 2048
        self.fc = nn.Linear(size, num_classes)

    def forward(self, x):
        x = self.r50_features(x)
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        output = self.fc(x)
        return output


class HPAMultiResNet50(nn.Module):

    def __init__(self, num_classes):
        super(HPAMultiResNet50, self).__init__()

        self.r50_red = resnet50(pretrained=True)
        self.r50_green = resnet50(pretrained=True)
        self.r50_blue = resnet50(pretrained=True)
        self.r50_yellow = resnet50(pretrained=True)

        def create_features(r50):
            return nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False),
                r50.bn1,
                r50.relu,
                r50.maxpool,
                r50.layer1,
                r50.layer2,
                r50.layer3,
                r50.layer4
            )

        self.r50_features_red = create_features(self.r50_red)
        self.r50_features_green = create_features(self.r50_green)
        self.r50_features_blue = create_features(self.r50_blue)
        self.r50_features_yellow = create_features(self.r50_yellow)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        size = 2048 * 4
        self.fc = nn.Linear(size, num_classes)

    def forward(self, x):

        x1, x2, x3, x4 = x[:, 0, :, :].unsqueeze(dim=1), \
                         x[:, 1, :, :].unsqueeze(dim=1), \
                         x[:, 2, :, :].unsqueeze(dim=1), \
                         x[:, 3, :, :].unsqueeze(dim=1)

        x1 = self.r50_features_red(x1)
        x2 = self.r50_features_red(x2)
        x3 = self.r50_features_red(x3)
        x4 = self.r50_features_red(x4)

        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x3 = self.avgpool(x3)
        x4 = self.avgpool(x4)

        x1 = x1.reshape((x1.shape[0], -1))
        x2 = x2.reshape((x2.shape[0], -1))
        x3 = x3.reshape((x3.shape[0], -1))
        x4 = x4.reshape((x4.shape[0], -1))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        output = self.fc(x)
        return output


if __name__ == "__main__":

    x = torch.rand((8, 4, 224, 224))

    model = HPAMultiResNet50(num_classes=28)
    y = model(x)
    y = torch.round(torch.sigmoid(y))
    print(y.shape)
    print(torch.equal(y, y ** 2))
