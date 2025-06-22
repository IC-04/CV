import torch.nn as nn
import torch.nn.functional as F

class seblock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(c, c // r),nn.ReLU(inplace=True),nn.Linear(c // r, c),nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.squeeze(x).view(b, c)
        weights = self.excitation(weights).view(b, c, 1, 1)
        return x * weights.expand_as(x)

class sebasicblock(nn.Module):
    exp = 1

    def __init__(self, inpl, pl, stride=1, r=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inpl, pl, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(pl)
        self.conv2 = nn.Conv2d(pl, pl, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(pl)
        self.se = seblock(pl, r)

        self.shortcut = nn.Sequential()
        if stride != 1 or inpl != self.exp*pl:
            self.shortcut = nn.Sequential(nn.Conv2d(inpl, self.exp*pl,kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(self.exp*pl))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

class seresnet(nn.Module):
    def __init__(self, block, num_blocks, num_c=10):
        super().__init__()
        self.inpl = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512*block.exp, num_c)

    def _make_layer(self, block, pl, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inpl, pl, stride))
            self.inpl = pl * block.exp
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def seresnet18():
    return seresnet(sebasicblock, [2,2,2,2])
