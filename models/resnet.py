import attr
import skorch
import torch
import torchvision
import random
import numpy as np

from sklearn.metrics import accuracy_score


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class BasicBlock(torch.nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.relu = torch.nn.ReLU(inplace=True)

        if downsample:
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, bias=False)
            bn = torch.nn.BatchNorm2d(out_channels)
            downsample = torch.nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)
        return x


class Bottleneck(torch.nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                     stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.conv3 = torch.nn.Conv2d(
            out_channels, self.expansion * out_channels, kernel_size=1,
            stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = torch.nn.ReLU(inplace=True)

        if downsample:
            conv = torch.nn.Conv2d(
                in_channels, self.expansion * out_channels, kernel_size=1,
                stride=stride, bias=False)
            bn = torch.nn.BatchNorm2d(self.expansion * out_channels)
            downsample = torch.nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class ResNet(torch.nn.Module):
    def __init__(self, input_dim, config, output_dim, preprocess=None):
        super().__init__()

        block = config.block
        n_blocks = config.n_blocks
        channels = config.channels

        assert len(n_blocks) == len(channels) == 4

        self.preprocess = preprocess or torch.nn.Identity
        self.in_channels = config.channels[0]
        self.conv1 = torch.nn.Conv2d(
            3, self.in_channels,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._layer(block, n_blocks[0], channels[0])
        self.layer2 = self._layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self._layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self._layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.in_channels, output_dim)

    def _layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)

        # Original ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x


@attr.s
class ResNetConfig:
    block = attr.ib(default=BasicBlock)
    n_blocks = attr.ib(default=[2, 2, 2, 2])
    channels = attr.ib(default=[64, 128, 256, 512])


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=x.shape)

        n_channels, width, height = x.shape
        conv = torch.nn.Conv2d(n_channels, 3, 1)
        net.set_params(module__preprocess=conv)

        module = net.module_

        n_pars = self.count_parameters(module)
        print(f"The model has {n_pars:,} trainable parameters")

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# For some reason num_workers doesn"t work with skorch :/
class DataIterator(torch.utils.data.DataLoader):
    def __init__(self, dataset, num_workers=4, *args, **kwargs):
        super().__init__(dataset, num_workers=num_workers, *args, **kwargs)


class VisionClassifierNet(skorch.NeuralNet):
    def predict(self, dataset):
        probas = self.predict_proba(dataset)
        return probas.argmax(-1)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(preds, y)


def build_model(batch_norm=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VisionClassifierNet(
        module=ResNet,
        module__input_dim=(1, 32, 32),
        module__config=ResNetConfig(),
        module__output_dim=10,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__param_groups=[
            ('features.*', {'lr': 5e-5}),
        ],
        optimizer__lr=5e-4,
        max_epochs=2,
        batch_size=256,
        iterator_train=DataIterator,
        iterator_train__shuffle=True,
        iterator_valid=DataIterator,
        iterator_valid__shuffle=False,
        device=device,
        callbacks=[
            ShapeSetter(),
        ]
    )
    return model


def main():
    # See https://pytorch.org/docs/stable/torchvision/models.html
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    print(f'Calculated means: {pretrained_means}')
    print(f'Calculated stds: {pretrained_stds}')

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(pretrained_size, padding=10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(device).fit(train)

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    print(model.predict(test))


if __name__ == "__main__":
    main()
