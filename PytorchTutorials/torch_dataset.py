"""
Practice of "Understanding torch with an example: a step-by-step tutorial"
from https://towardsdatascience.com/understanding-torch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
"""
import numpy as np
import torch
# The torchvision package consists of popular datasets, model architectures,
# and common image transformations for computer vision.
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchsummary


def data_generation():
    """
    Generate synthetic data for Regression.
    :return:
    """
    # Data Generation
    np.random.seed(11)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

    # Shuffle the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    # Split train and validation set.
    train_idx = idx[:80]
    val_idx = idx[80:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, y_train, x_val, y_val


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def split_data_torch():
    np.random.seed(11)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)

    dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [80, 20])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    return train_loader, val_loader


def make_train_step(model, loss_fn, optimizer):
    """
    Builds function that performs a step in the train loop
    :param model:
    :param loss_fn:
    :param optimizer:
    :return:
    """
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step


def regression_torch_style_dataset():
    x_train, y_train, x_val, y_val = data_generation()

    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)
    print(model.state_dict())

    lr = 1e-1
    n_epochs = 1000
    # # Here, tensors are CPU tensor now without .to(device)
    # # We don't want our whole training data to be loaded into GPU tensors, because it takes up
    # # space in our precious graphics card's RAM.
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # # Here, CustomDataset is equal to TensorDataset functionally.
    # train_data = CustomDataset(x_train_tensor, y_train_tensor)
    # print(train_data[0:10])
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    print(train_data[0:10])

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # # DataLoader behaves like an iterator, so we can loop over it and fetch a
    # # different mini-batch every time.
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # sample_batch = next(iter(train_loader))
    # sample_batch = iter(train_loader).next()
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            # # The dataset "lives" in the CPU, so do our mini-batches therefore, we
            # # need to send those mini-batches to the device where the model "lives".
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

    print(model.state_dict())


def regression_torch_style():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)
    # print(model) can provide information about this model.
    print("--------------- model summary by print(model)  ---------------")
    print(model)
    print("--------------- model summary by summary()  ---------------")
    torchsummary.summary(model, input_size=(1, 1))
    print(model.state_dict())

    lr = 1e-1
    n_epochs = 1000

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_step = make_train_step(model, loss_fn, optimizer)

    train_loader, val_loader = split_data_torch()
    losses, val_losses = [], []

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

        # It's a good practice to wrap the validation inner loop with torch.no_grad()
        # to disable any gradient calculation that you may inadvertently trigger, as
        # gradients should belong in training, not in validation steps;
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # Set the model to evaluation mode
                model.eval()

                y_hat = model(x_val)
                val_loss = loss_fn(y_val, y_hat)
                val_losses.append(val_loss.item())

    print(model.state_dict())

# Todo: make_dot() in package torchviz
# def show_make_dot():
#     import graphviz
#     from torch.autograd import Variable
#     from torchviz import make_dot, make_dot_from_trace
#
#     model = torch.nn.Sequential()
#     model.add_module('W0', torch.nn.Linear(8, 16))
#     model.add_module('tanh', torch.nn.Tanh())
#     model.add_module('W1', torch.nn.Linear(16, 1))
#
#     x = Variable(torch.randn(1, 8))
#     y = model(x)
#
#     dot = make_dot(y.mean(), params=dict(model.named_parameters()))
#     # dot.format = 'svg'
#     # dot.render()


def get_CIFAR10_dataset():
    """
    Load CIFAR10 dataset by torchvision and DataLoader
    :return:
    """
    # Download and construct CIFAR-10 dataset.
    train_dataset = torchvision.datasets.CIFAR10(root="./dataset/",
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # Fetch one data pair (read data from disk).
    image, label = train_dataset[0]
    print(image.size(), " -- ", label)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # # Get a sample mini-batch from DataLoader.
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # images, labels = next(data_iter)
    # print(images.size(), " -- ", labels.size())

    for images, labels in train_loader:
        # Training code should be written here.
        print(images.size(), " -- ", labels.size())


def main():
    torch.manual_seed(11)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # regression_torch_style_dataset()
    regression_torch_style()
    # get_CIFAR10_dataset()


if __name__ == '__main__':
    main()