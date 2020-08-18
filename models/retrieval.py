import torch
import skorch
import random
import torchvision
import numpy as np

from sklearn.neighbors import KDTree

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Embedding(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embeddings = torch.nn.Linear(512, 128)

    def forward(self, x):
        return self.embeddings(self.backbone(x))


def l2(a):
    return (a ** 2).sum(-1).view(-1, 1)


def dist(a, b):
    return -2 * a @ b.T + l2(a) + l2(b)


class RetrievalLoss(torch.nn.Module):
    def __init__(self, sim=dist, delta=1.0):
        super().__init__()
        self.delta = delta
        self.sim = sim

    def forward(self, queries, targets):
        with torch.no_grad():
            distances = self.sim(queries, queries)

            # exploit the broadcasting
            same_idx = targets.view(-1, 1) == targets.view(1, -1)
            pos = queries[(distances * same_idx).argmax(-1)]

            neg_idx = (distances * ~same_idx).argmax(-1)
            neg = queries[neg_idx]

        loss = self.delta - l2(queries - pos) + l2(queries - neg)
        return torch.nn.functional.relu(loss).mean()


def accuracy_at_k(y, X, K, sample=None):
    kdtree = KDTree(X)
    y_true = y[:sample]

    indices_of_neighbours = kdtree.query(
        X[:sample], k=K + 1, return_distance=False)[:, 1:]

    y_hat = y[indices_of_neighbours]

    matching_category_mask = np.expand_dims(np.array(y_true), -1) == y_hat
    matching_cnt = np.sum(matching_category_mask.sum(-1) > 0)
    accuracy = matching_cnt / len(y_true)
    return accuracy


def score(name, K=5):
    def f(model, X, y):
        return accuracy_at_k(y, model.predict(X), K=K)
    f.__name__ = f"{name} acc@{K}"
    return f


def build_model():
    resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet18.fc = torch.nn.Identity()

    model = skorch.NeuralNet(
        module=Embedding,
        module__backbone=resnet18,
        optimizer=torch.optim.SGD,
        lr=0.0005,
        batch_size=512,
        max_epochs=2,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        criterion=RetrievalLoss,
        callbacks=[
            skorch.callbacks.EpochScoring(score("valid"), False),
            # skorch.callbacks.EpochScoring(score("train"), False, True),
        ]
    )

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    traint = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(pretrained_size, padding=10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    testt = torchvision.transforms.Compose([
        torchvision.transforms.Resize(pretrained_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    return model, traint, testt


def main():
    model, traint, testt = build_model()

    train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=traint,
    )
    model = build_model().fit(train)

    test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=testt,
    )
    preds = model.predict(test)
    print(preds)


if __name__ == '__main__':
    main()
