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

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),  # Original size
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
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x


def from_pretrained(module, pretrained_module):
    custom = module.state_dict()
    pretrained = pretrained_module.state_dict()
    # Add weights from the new module missing in the old model
    new_layers = {
        k: v for k, v in custom.items()
        if k not in pretrained or "classifier" in k
    }
    pretrained.update(new_layers)
    module.load_state_dict(pretrained)


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=x.shape)

        module = net.module_

        pretrained = torchvision.models.vgg11_bn(pretrained=True)
        from_pretrained(module, pretrained)

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
        torchvision.transforms.RandomCrop(32, padding=2),
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
