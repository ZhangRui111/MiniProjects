"""
How to visualize: tensorboard --logdir="tb" --port 6006
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from utils.logger import Logger


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def main():
    global dev
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 3
    batch_size = 64
    learning_rate = 0.001

    train_dataset = torchvision.datasets.MNIST(root="./data/",
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger = Logger("./logs/tb")

    model = NeuralNet(input_size, hidden_size, num_classes).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(dev).view(-1, 28*28)
            labels = labels.to(dev)

            model.train()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Training accuracy
            with torch.no_grad():
                model.eval()
                _, pred = torch.max(output, dim=1)
                acc = (pred.squeeze() == labels.squeeze()).float().mean()

            if (step + 1) % 10 == 0:
                print("Epoch-Step [{}/{}]-[{}], Loss: {:.4f}, Acc: {:.2f}"
                      .format(epoch, num_epochs, step, loss.item(), acc.item()))

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {'loss': loss.item(), 'acc': acc.item()}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag, value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary)
                info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

                for tag, value in info.items():
                    logger.image_summary(tag, value, step)

            step += 1


if __name__ == "__main__":
    main()
