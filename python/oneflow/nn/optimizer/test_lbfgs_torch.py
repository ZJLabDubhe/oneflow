import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
# from lbfgs import Lbfgs
# from lbfgs_torch import LBFGS
from torch import optim
import time

BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_dataloader = torch.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_data, BATCH_SIZE, shuffle=False
)

for x, y in train_dataloader:
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    break


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(DEVICE)
print(model)

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
# optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

optimizer = optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", lr=3e-3, max_iter=10)
# optimizer = LBFGS(model.parameters(), line_search_fn="strong_wolfe", lr=3e-3, max_iter=10)


def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # optimizer.step()

        def closure():
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            # print('loss')
            # print(loss)
            # time.sleep(2)
            return loss

        optimizer.step(closure)

        current = batch * BATCH_SIZE
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(iter, model, loss_fn):
    size = len(iter.dataset)
    num_batches = len(iter)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            test_loss += loss_fn(pred, y)
            # bool_value = (pred.argmax(1).to(dtype=torch.int64)==y)
            # correct += float(bool_value.sum().numpy())

            pred = pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()

    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    # correct /= (float(size))
    cor = float(correct) / float(size)
    print("Avg loss: {test_loss:>8f}")
    print("correct:", cor * 100, "%")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
