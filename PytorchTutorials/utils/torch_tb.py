"""
This is more preferred, compared with torch_tb_logger.py
How to visualize: tensorboard --logdir="tb" --port 6006
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


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

    train_dataset = torchvision.datasets.MNIST(root="./data/MNIST",
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter("./logs/tb/")

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

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log model graph (only once in one log file.)
            if step == 0:
                writer.add_graph(model, images)

            if (step + 1) % 10 == 0:
                print("Epoch-Step [{}/{}]-[{}], Loss: {:.4f}, Acc: {:.2f}"
                      .format(epoch, num_epochs, step, loss.item(), acc.item()))

                # 2. Log scalar values (scalar summary)
                info = {'loss': loss.item(), 'acc': acc.item()}

                for tag, value in info.items():
                    writer.add_scalar(tag, value, step)

                # 3. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), step)
                    writer.add_histogram(tag, value.grad.data.cpu().numpy(), step)

                # 4. Log training images (image summary)
                input_images = images[:9].reshape(-1, 1, 28, 28)
                # (1) separate images
                # for idx in range(input_images.shape[0]):
                #     writer.add_image(str(idx), input_images[idx], step)
                # (2) merged images
                grid = torchvision.utils.make_grid(input_images, nrow=3, padding=3, pad_value=1)
                writer.add_image('images', grid, step)

            step += 1


if __name__ == "__main__":
    main()
