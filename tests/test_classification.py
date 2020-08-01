import pytest
import torchvision

from models.mlp import build_model as build_mlp
from models.lenet import build_model as build_lenet
from models.alexnet import build_model as build_alexnet
from models.vgg import build_model as build_vgg


class DownsampledMNIST(torchvision.datasets.MNIST):
    def __init__(self, n_samples=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples


@pytest.fixture
def data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.), (1.))
    ])

    data = DownsampledMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    return data


@pytest.mark.parametrize("build", [
    build_mlp,
    build_lenet,
    build_alexnet,
    build_vgg,
])
def test_classifier(build, data):
    model = build().fit(data)
    y = [data[i][1] for i in range(len(data))]

    assert model.score(data, y) >= (1. / 10)
