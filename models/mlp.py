import skorch
import torch
import torchvision
import random
import numpy as np


from operator import mul
from functools import reduce


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = torch.nn.Linear(input_dim, 250)
        self.hidden_fc = torch.nn.Linear(250, 100)
        self.output_fc = torch.nn.Linear(100, output_dim)

    def forward(self, x):
        # x = [batch_size, height, width]
        batch_size = x.shape[0]
        # x = [batch_size, height * width]
        x = x.view(batch_size, -1)

        # h_1 = [batch_size, 250]
        h_1 = torch.nn.functional.relu(self.input_fc(x))

        # h_2 = [batch_size, 100]
        h_2 = torch.nn.functional.relu(self.hidden_fc(h_1))

        # y_pred = [batch_size, output_dim]
        return self.output_fc(h_2)


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__input_dim=reduce(mul, x.shape, 1))
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
        module=MLP,
        module__input_dim=100,
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
        ]
    )
    return model


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.), (1.))
    ])

    train = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(device).fit(train)

    test = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    print(model.predict(test))


if __name__ == "__main__":
    main()
