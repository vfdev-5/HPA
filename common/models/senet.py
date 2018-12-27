
import torch
import torch.nn as nn

from pretrainedmodels.models.senet import SENet, SEResNeXtBottleneck


class HPASENet50(nn.Module):

    def __init__(self, num_classes):
        super(HPASENet50, self).__init__()

        self.senet = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                           dropout_p=None, inplanes=64, input_3x3=False,
                           downsample_kernel_size=1, downsample_padding=0,
                           num_classes=num_classes)
        
        self.senet.layer0[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        size = 2048
        self.fc = nn.Linear(size, num_classes)

    def forward(self, x):
        x = self.senet.features(x)
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        output = self.fc(x)
        return output


if __name__ == "__main__":

    x = torch.rand((8, 4, 420, 420))

    model = HPASENet50(num_classes=28)
    y = model(x)
    y = torch.round(torch.sigmoid(y))
    print(y.shape)
    print(torch.equal(y, y ** 2))
