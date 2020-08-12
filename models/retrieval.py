import torch
import skorch
import torchvision


class Embedding(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embeddings = torch.nn.Linear(512, 128)

    def forward(self, x):
        return self.embeddings(self.backbone(x))


def cosine(a, b):
    unit_a = a / a.norm(p=2, dim=-1, keepdim=True)
    unit_b = b / b.norm(p=2, dim=-1, keepdim=True)
    return (unit_a * unit_b).sum(-1)


class RetrievalLoss(torch.nn.Module):
    def __init__(self, sim=cosine, delta=1.0):
        super().__init__()
        self.delta = delta
        self.sim = sim

    def forward(self, queries, targets):
        wrong = self.negatives(queries, targets)
        correct = targets
        return torch.nn.functional.relu(
            self.delta - self.sim(queries, correct) + self.sim(queries, wrong)
        ).mean()

    def negatives(self, a, b):
        with torch.no_grad():
            sim = self.sim(b.unsqueeze(0), b.unsqueeze(1))
            return b[(sim - torch.eye(*sim.shape)).argmax(0)]


def build_model():
    resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet18.fc = torch.nn.Identity()

    model = skorch.NeuralNet(
        module=Embedding,
        module__backbone=resnet18,
        batch_size=512,
        criterion=RetrievalLoss,
    )
    return model


def main():
    pass


if __name__ == '__main__':
    main()
