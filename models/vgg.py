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


def count_output_size(model, shape, batch_size=2):
    with torch.no_grad():
        batch = torch.rand(batch_size, *shape)
        return model(batch).data.view(batch_size, -1).shape[-1]


VGG_CONFIG = {
    "vgg_mnist": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',
              512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def prepare_vgg_layers(config, batch_norm, in_channels=3):
    layers = []
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(c),
                           torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = c

    return torch.nn.Sequential(*layers)


class VGG(torch.nn.Module):
    def __init__(self, input_dim, output_dim, version="vgg_mnist",
                 batch_norm=False):
        super().__init__()

        self.preprocess = torch.nn.Identity()

        n_channes = input_dim[0]
        if n_channes != 3:
            self.preprocess = torch.nn.Conv2d(n_channes, 3, kernel_size=1)

        self.features = prepare_vgg_layers(VGG_CONFIG[version], batch_norm)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(7)

        fc_size = count_output_size(
            torch.nn.Sequential(self.preprocess, self.features),
            input_dim
        )

        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear(512 * 7 * 7, 4096), # Original size
            torch.nn.Linear(fc_size, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=x.shape)

        # Load the pretrained model
        # pretrained = torchvision.models.vgg11_bn(pretrained=True)
        # net.module_.load_state_dict(pretrained.state_dict())

        n_pars = self.count_parameters(net.module_)
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


def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(
            m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(m.bias.data, 0)


def build_model(version="vgg_mnist", batch_norm=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VisionClassifierNet(
        module=VGG,
        module__input_dim=(1, 32, 32),
        module__output_dim=10,
        module__version=version,
        module__batch_norm=batch_norm,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        max_epochs=2,
        batch_size=256,
        iterator_train=DataIterator,
        iterator_train__shuffle=True,
        iterator_valid=DataIterator,
        iterator_valid__shuffle=False,
        device=device,
        callbacks=[
            ShapeSetter(),
            skorch.callbacks.Initializer('*', fn=initialize_weights),
        ]
    )
    return model


def main():
    # It's ugly to download dataset two times to extract the stats.
    # TODO: Fix this later
    train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
    )

    means = train.data.mean(axis=(0, 1, 2)) / 255
    stds = train.data.std(axis=(0, 1, 2)) / 255

    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(32, padding=2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
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
