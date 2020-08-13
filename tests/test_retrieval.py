import pytest
import torchvision
import numpy as np

from functools import partial
from models.retrieval import build_model, accuracy_at_k


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, n_samples=100, *args, **kwargs):
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
    model, traint, testt = build_model()
    dataset = data(transform=traint)
    model.fit(dataset)

    test = data(transform=testt)
    embs = model.predict(test)
    y = np.array([test[i][-1] for i in range(len(test))])
    acc5 = accuracy_at_k(y, embs, 5)
    print(acc5)
    assert acc5 > 0.1
