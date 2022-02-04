import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class OdomNet(nn.Module):

    def __init__(self):
        super(OdomNet, self).__init__()
        self.featuresImage = nn.Sequential(models.resnet18(pretrained=True))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.featuresIMU = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

        self.regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 + 10, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )
    def forward(self, x1,x2):
        x1 = self.featuresImage(x1.float())
        x2 = self.featuresIMU(x2.float())
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.regression(x)
        return x