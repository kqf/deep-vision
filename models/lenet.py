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


class LeNet(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5)

        self.fc_1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = torch.nn.Linear(120, 84)
        self.fc_3 = torch.nn.Linear(84, output_dim)

    def forward(self, x):
        # [batch_size, 1, 28, 28] -> [batch_size, 6, 24, 24]
        x = self.conv1(x)

        # [batch_size, 6, 12, 12]
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)

        x = torch.nn.functional.relu(x)

        # [batch_size, 16, 8, 8]
        x = self.conv2(x)

        # [batch_size, 16, 4, 4]
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)

        x = torch.nn.functional.relu(x)

        # [batch_size, 16 * 4 * 4 = 256]
        x = x.view(x.shape[0], -1)

        # [batch_size, 120]
        x = self.fc_1(x)
        x = torch.nn.functional.relu(x)

        # x = batch_size, 84]
        x = self.fc_2(x)
        x = torch.nn.functional.relu(x)

        # [batch_size, output dim]
        x = self.fc_3(x)
        return x


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
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


def build_model(device=torch.device("cpu")):
    best_checkpoint = skorch.callbacks.Checkpoint(
        monitor='valid_loss_best',
        dirname="lenet.checkpoint"
    )

    model = VisionClassifierNet(
        module=LeNet,
        module__output_dim=10,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        max_epochs=2,
        batch_size=64,
        iterator_train=DataIterator,
        iterator_train__shuffle=True,
        iterator_valid=DataIterator,
        iterator_valid__shuffle=False,
        device=device,
        callbacks=[
            ShapeSetter(),
            best_checkpoint,
            skorch.callbacks.LoadInitState(best_checkpoint),
        ]
    )
    return model


def main():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(5, fill=(0,)),
        torchvision.transforms.RandomCrop(28, padding=2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.), (1.))
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.), (1.))
    ])

    train = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(device).fit(train)

    test = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    print(model.predict(test))


if __name__ == "__main__":
    main()
