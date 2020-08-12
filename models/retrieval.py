import torch


class Backbone(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


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
