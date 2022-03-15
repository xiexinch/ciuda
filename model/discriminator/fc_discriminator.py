import torch.nn as nn


class FCDiscriminator(nn.Module):

    def __init__(self, in_channels, base_channels=64, num_convs=4):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               base_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d(base_channels,
                               base_channels * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2,
                               base_channels * 4,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4,
                               base_channels * 4,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        self.classifier = nn.Conv2d(base_channels * 4,
                                    1,
                                    kernel_size=4,
                                    stride=1,
                                    padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
