import pytest
import torchvision

from models.mlp import build_model


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


def test_mlp(data):
    build_model().fit(data)
