import attr
import skorch
import torch
import torchvision
import random
import operator
import numpy as np

from sklearn.metrics import accuracy_score


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def conv(*args, **kwargs):
    return torch.nn.Conv2d(*args, bias=False, **kwargs)


def skip(cin, cout, stride):
    cnv = conv(cin, cout, kernel_size=1, stride=stride)
    bn = torch.nn.BatchNorm2d(cout)
    return torch.nn.Sequential(cnv, bn)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=False):
        super().__init__()

        self.conv1 = conv(cin, cout, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(cout)
        self.conv2 = conv(cout, cout, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(cout)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = torch.nn.Identity() if not downsample else skip(
            cin, cout, stride)

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        i = self.downsample(i)

        x += i
        x = self.relu(x)
        return x


class Bottleneck(torch.nn.Module):

    expansion = 4

    def __init__(self, cin, cout, stride=1, downsample=False):
        super().__init__()

        self.conv1 = conv(cin, cout, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(cout)
        self.conv2 = conv(cout, cout, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(cout)
        self.conv3 = conv(cout, self.expansion * cout, kernel_size=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion * cout)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = torch.nn.Identity() if not downsample else skip(
            cin, self.expansion * cout, stride)

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

        i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class ResNet(torch.nn.Module):
    def __init__(self, input_dim, config, output_dim, preprocess=None):
        super().__init__()

        self.config = config
        block = config.block
        n_blocks = config.n_blocks
        channels = config.channels

        assert len(n_blocks) == len(channels) == 4

        self.preprocess = preprocess or torch.nn.Identity
        self.cin = config.channels[0]

        self.conv1 = conv(3, self.cin, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(self.cin)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._layer(block, n_blocks[0], channels[0])
        self.layer2 = self._layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self._layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self._layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.cin, output_dim)

    def _layer(self, block, n_blocks, channels, stride=1):
        downsample = self.cin != block.expansion * channels
        layers = [block(self.cin, channels, stride, downsample)]

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.cin = block.expansion * channels
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


# Default values are for ResNet18
@attr.s
class ResNetConfig:
    block = attr.ib(default=BasicBlock)
    n_blocks = attr.ib(default=[2, 2, 2, 2])
    channels = attr.ib(default=[64, 128, 256, 512])


CONFIGURATIONS = {
    "resnet18": ResNetConfig(),
    "resnet50": ResNetConfig(
        block=Bottleneck,
        n_blocks=[3, 4, 6, 3],
        channels=[64, 128, 256, 512]
    )
}


def from_pretrained(module, pretrained_module):
    custom = module.state_dict()
    pretrained = pretrained_module.state_dict()
    # Add weights from the new module missing in the old model
    new_layers = {
        k: v for k, v in custom.items()
        if k not in pretrained or "fc" in k
    }
    pretrained.update(new_layers)
    module.load_state_dict(pretrained)


class ShapeSetter(skorch.callbacks.Callback):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=x.shape)

        n_channels, width, height = x.shape
        fix_num_channels = conv(n_channels, 3, kernel_size=1)
        net.set_params(module__preprocess=fix_num_channels)

        module = net.module_

        load = operator.methodcaller(self.config, pretrained=True)
        from_pretrained(module, load(torchvision.models))

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


def build_model(config="resnet50", batch_norm=True, lr=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VisionClassifierNet(
        module=ResNet,
        module__input_dim=(1, 32, 32),
        module__config=CONFIGURATIONS[config],
        module__output_dim=10,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__param_groups=[
            ('conv1.*', {'lr': lr / 10}),
            ('bn1.*', {'lr': lr / 10}),
            ('layer1.*', {'lr': lr / 8}),
            ('layer2.*', {'lr': lr / 6}),
            ('layer3.*', {'lr': lr / 4}),
            ('layer4.*', {'lr': lr / 2}),
        ],
        optimizer__lr=lr,
        max_epochs=2,
        batch_size=256,
        iterator_train=DataIterator,
        iterator_train__shuffle=True,
        iterator_valid=DataIterator,
        iterator_valid__shuffle=False,
        device=device,
        callbacks=[
            ShapeSetter(config),
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
