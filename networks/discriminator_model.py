import torch
import torch.nn as nn

"""PatchGAN discriminator, as described in Pix2Pix paper"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, k_size, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, k_size=(4,4), features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=k_size,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, k_size, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, k_size, stride=1, padding=1, padding_mode="reflect"
            ), # no need to add a sigmoid layer here, since training uses BCEWithLogits loss
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


class PerPixelDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2),
        )

        self.model = nn.Sequential(
            nn.Conv2d(features[0],
                      features[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False,
                      ),
            nn.BatchNorm2d(features[1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[1],
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      )
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    """
    Classical discriminator that returns a scalar value. To be used on LSGAN
    """
    def __init__(self, in_channels=1, k_size=(4,4), features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=k_size,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, k_size, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.extend([
            nn.Conv2d(in_channels, 1, k_size, stride=1, padding=1, padding_mode="reflect"),
            nn.MaxPool2d(kernel_size=2,)]
        )

        self.fc = nn.Linear(15*15, 1)
        self.model = nn.Sequential(*layers)


    def forward(self, x, y):
        
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # -> [B,1]


def test():
    x = torch.randn((1, 1, 257, 257))
    y = torch.randn((1, 1, 257, 257))
    model = Discriminator(in_channels=1, k_size=(4,4))
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    test()