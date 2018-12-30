
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34


class HPAAutoEncodeResNet34(nn.Module):

    def __init__(self, num_classes):
        super(HPAAutoEncodeResNet34, self).__init__()

        self.resnet = resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.stem = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )

        self.head4 = self.create_head(512, 256)
        self.head3 = self.create_head(256, 128)
        self.head2 = self.create_head(128, 64)
        self.head1 = self.create_head(64, 64)
        # self.head1b = self.create_head(64, 64)
        self.last = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def create_head(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    # def decode(self, c1, c2, c3, c4, c5):
    #     p4 = self.head4(c5) + c4
    #     p3 = self.head3(p4) + c3
    #     p2 = self.head2(p3) + c2
    #     p1 = self.head1(p2) + self.head1b(c1)
    #     return p1

    def forward(self, x):

        c1 = self.stem(x)

        c2 = self.resnet.layer1(c1)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        y = self.avgpool(c5)
        y = y.reshape((y.shape[0], -1))
        y = self.fc(y)

        if self.training:
            # p1 = self.decode(c1, c2, c3, c4, c5)
            p4 = self.head4(c5)
            p3 = self.head3(p4)
            p2 = self.head2(p3)
            p1 = self.head1(p2)
            output = self.last(p1)
            return output, y

        return y


if __name__ == "__main__":

    x = torch.rand((8, 4, 320, 320))

    model = HPAAutoEncodeResNet34(num_classes=28)
    out, y = model(x)
    print(out.shape, y.shape)
