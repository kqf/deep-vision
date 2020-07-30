import skorch
import torch
import torchvision
import random
import numpy as np


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def count_output_size(model, shape):
    with torch.no_grad():
        return model(torch.rand(1, *shape)).data.view(1, -1).shape[-1]


class AlexNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        n_channels, width, height = input_dim
        self.features = torch.nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            torch.nn.Conv2d(n_channels, 64, 3, 2, 1),
            torch.nn.MaxPool2d(2),  # kernel_size
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 192, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 384, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True)
        )

        fc_size = count_output_size(self.features, input_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fc_size, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=x.shape)
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


def build_model(device=torch.device("cpu")):
    model = VisionClassifierNet(
        module=AlexNet,
        module__input_dim=(1, 32, 32),
        module__output_dim=10,
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
