"""
KeyInfo about ResNet:

The convolutional layers in ResNet mostly have 3×3 filters and follow two
simple design rules:
(i) for the same output feature map size, the layers have the same number
of filters, e.g., [100, 32, 16, 16] -> [100, 32, 16, 16] (B, C, W, H) and
(ii) if the feature map size is halved, the number of  filters is doubled
so as to preserve the time complexity per layer, e.g., [100, 32, 16, 16] ->
[100, 64, 8, 8].

ResNet perform downsampling directly by convolutional layers that have a
stride of 2.
"""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#  Update the optimizer's learning rate.
def update_lr(optimizer, lr):
    # An optimizer can have multiple param groups, this is very useful when
    # one wants to specify per-layer learning rates. e.g.,
    # optim.SGD([{'params': model.base.parameters()},
    #            {'params': model.classifier.parameters(), 'lr': 1e-3}],
    #            lr=1e-2, momentum=0.9)
    # This means that model.base’s parameters will use the default learning
    # rate of 1e-2, model.classifier’s parameters will use a learning rate of
    # 1e-3, and a momentum of 0.9 will be used for all parameters.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample:
            residual = self.downsample(x)
        output += residual
        output = self.relu2(output)
        return output


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.cur_in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_block, stride=1):
        # Whether do downsampling
        downsample = None
        if (stride != 1) or (self.cur_in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.cur_in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        # Stack multiple Residual unit in one layer.
        layers = [block(self.cur_in_channels, out_channels, stride, downsample)]
        self.cur_in_channels = out_channels
        for i in range(1, num_block):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        # * means "Unpacking", e.g.:
        # def accept(a, b, c):
        #     print(a + b + c)
        # a = [1, 2, 3]
        # accept(*a)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)  # [B, 16, 32, 32]
        output = self.layer1(output)  # [B, 16, 32, 32], stride==1, thus no downsampling
        output = self.layer2(output)  # [B, 32, 16, 16]
        output = self.layer3(output)  # [B, 64, 8, 8]
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def main():
    global dev
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 80
    learning_rate = 0.001
    batch_size = 128

    train_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10/",
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10/",
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Image preprocessing modules
    # .Compose(): Composes several transforms in a list together
    transform = transforms.Compose([
        transforms.Pad(padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32),
        transforms.ToTensor()]
    )

    model = ResNet(ResidualBlock, [2, 2, 2]).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cur_lr = learning_rate
    step = 0
    writer = SummaryWriter("./logs/resnet/")

    s_time = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(dev)
            labels = labels.to(dev)

            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step == 0:
                writer.add_graph(model, images)

            if (step + 1) % 100 == 0:
                writer.add_scalar('loss', loss.item(), step)

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), step)
                    writer.add_histogram(tag, value.grad.data.cpu().numpy(), step)

                print("Epoch-Step {}/{}-{} | Loss {}".format(epoch, num_epochs, step, loss.item()))

            step += 1

        if (epoch + 1) % 20 == 0:
            cur_lr = cur_lr / 3
            update_lr(optimizer, cur_lr)

    e_time = time.time()
    train_time = e_time - s_time
    print("Training takes {}s".format(train_time))

    # model.load_state_dict(torch.load("./logs/resnet/params.ckpt"))
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        model.eval()
        for images, labels in test_loader:
            images = images.to(dev)
            labels = labels.to(dev)

            output = model(images)
            _, pred = torch.max(output, dim=1)
            correct_num += (pred.squeeze() == labels.squeeze()).sum()
            total_num += labels.size(0)
        print("Accuracy is {} %".format(100 * correct_num / total_num))

    torch.save(model.state_dict(), "./logs/resnet/params.ckpt")
    writer.close()


if __name__ == "__main__":
    main()
