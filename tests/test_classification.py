import pytest
import torchvision

from models.mlp import build_model as build_mlp
from models.lenet import build_model as build_lenet
from models.alexnet import build_model as build_alexnet


@pytest.fixture
def data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.), (1.))
    ])

    data = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    return data


@pytest.mark.parametrize("build", [
    # build_mlp,
    # build_lenet,
    build_alexnet
])
def test_classifier(build, data):
    build().fit(data)
