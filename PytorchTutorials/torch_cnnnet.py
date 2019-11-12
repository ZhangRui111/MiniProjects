import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class CNNNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, output_size)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def main():
    global dev
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 3
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001

    LOAD_MODEL = True
    LOAD_AS_PARAMS = True
    SAVE_MODEL = True
    SAVE_AS_PARAMS = True

    train_dataset = torchvision.datasets.MNIST(root="./data/",
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data/",
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter("./logs/cnn/")
    step = 0

    if LOAD_MODEL is True:
        if LOAD_AS_PARAMS is True:
            model = CNNNet(num_classes).to(dev)
            model.load_state_dict(torch.load("./logs/cnn/params.ckpt"))
        else:
            model = torch.load("./logs/cnn/model.ckpt")

    else:
        model = CNNNet(num_classes).to(dev)
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(dev)
                labels = labels.to(dev)

                model.train()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()

                with torch.no_grad():
                    model.eval()
                    _, pred = torch.max(output, 1)
                    acc = (pred.squeeze() == labels.squeeze()).float().mean()

                if step == 0:
                    writer.add_graph(model, images)

                if (step + 1) % 10 == 0:
                    print("Epoch-Step {}/{}--{} | Loss {} | Accuracy {}".format(
                        epoch, num_epochs, step, loss.item(), acc.item()))

                    info = {'loss': loss.item(), 'accuracy': acc.item()}

                    for tag, value in info.items():
                        writer.add_scalar(tag, value, step)

                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), step)
                        writer.add_histogram(tag, value.grad.data.cpu().numpy(), step)

                step += 1

    if SAVE_MODEL:
        if SAVE_AS_PARAMS is True:
            torch.save(model.state_dict(), "./logs/cnn/params.ckpt")
        else:
            torch.save(model, "./logs/cnn/model.ckpt")

    writer.close()


if __name__ == "__main__":
    main()
