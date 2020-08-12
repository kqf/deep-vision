import pytest
import torchvision

from functools import partial
from models.retrieval import build_model


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, n_samples=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples


@pytest.fixture(scope="module")
def data():
    data = partial(
        DownsampledCIFAR10,
        root="./data",
        train=True,
        download=True,
    )
    return data


def test_retriever(data):
    model, traint, _ = build_model()
    dataset = data(transform=traint)
    model.fit(dataset)

    print(model)
