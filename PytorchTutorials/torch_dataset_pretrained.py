"""
Practice of "Understanding torch with an example: a step-by-step tutorial"
from https://towardsdatascience.com/understanding-torch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
from torchvision.transforms import transforms
import torchsummary


def data_generation():
    """
    Generate synthetic data for Regression.
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


def torchvision_ops():
    """
    torchvision involves torchvision.ops.~
    """
    # torchvision.ops.nms()
    # torchvision.ops.roi_pool()
    # torchvision.ops.roi_align()
    pass


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


def get_MNIST_dataset():
    train_dataset = torchvision.datasets.MNIST(root="./data/MNIST/",
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data/MNIST/",
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Traversal search by two ways.
    for images, labels in train_loader:
        print(images.size(), " -- ", labels.size())

    for i, (images, labels) in enumerate(train_loader):
        print(i, images.size(), " -- ", labels.size())


def get_CIFAR10_dataset():
    """ Download and construct CIFAR-10 dataset. """
    # ToTensor() works for the image, whose elements are in range 0 to 255.
    # It converts data in the range 0-255 to 0-1.
    train_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10/",
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10/",
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # Fetch one data pair (read data from disk).
    image, label = train_dataset[0]
    print(image.size(), " -- ", label)
    # # Get a sample mini-batch from DataLoader.
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # images, labels = next(data_iter)
    # print(images.size(), " -- ", labels.size())

    # Traversal search by two ways.
    for images, labels in train_loader:
        print(images.size(), " -- ", labels.size())

    for i, (images, labels) in enumerate(train_loader):
        print(i, images.size(), " -- ", labels.size())


def get_CoCoDetection():
    train_path2data = "./data/COCO/train2017/"
    train_path2json = "./data/COCO/annotations/instances_train2017.json"
    val_path2data = "./data/COCO/val2017/"
    val_path2json = "./data/COCO/annotations/instances_val2017.json"

    coco_train = torchvision.datasets.CocoDetection(root=train_path2data,
                                                    annFile=train_path2json,
                                                    transform=None)
    coco_val = torchvision.datasets.CocoDetection(root=val_path2data,
                                                  annFile=val_path2json,
                                                  transform=transforms.ToTensor())
    coco_train_loader = DataLoader(coco_train, batch_size=16, shuffle=True)
    coco_val_loader = DataLoader(coco_val, batch_size=16, shuffle=True)

    print(coco_train[0])


def get_VocDetection():
    years = ['2007', '2012']
    sets = ['train', 'trainval', 'val']
    datasets, loaders = [], []

    # for year in years:
    #     for set in sets:
    #         print("{} - {}".format(year, set))
    #         dataset = torchvision.datasets.VOCDetection("./data/VOC/",
    #                                                     year=year,
    #                                                     image_set=set,
    #                                                     download=True,
    #                                                     transform=None)
    #         loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #         datasets.append(dataset)
    #         loaders.append(loader)

    dataset = torchvision.datasets.VOCDetection("./data/VOC/",
                                                year='2007',
                                                image_set='train',
                                                download=True,
                                                transform=None)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(dataset[0])


def pretrained_model():
    resnet34 = torchvision.models.resnet34(pretrained=True).to(device)
    # Finetune only the top layer of the model.
    for param in resnet34.parameters():
        param.requires_grad = False
    # Replace the top layer for finetuning.
    # in_features: size of each input sample
    resnet34.fc = torch.nn.Linear(resnet34.fc.in_features, 100)
    # Forward pass.
    images = torch.randn(64, 3, 224, 224)
    outputs = resnet34(images)
    print(outputs.size())


def save_load_model():
    resnet34 = torchvision.models.resnet34(pretrained=True).to(device)
    # print(resnet34)
    # torchsummary.summary(resnet34, input_size=(3, 224, 224))

    # Save and load the entire model.
    torch.save(resnet34, "./logs/model/model.ckpt")
    load_model = torch.load("./logs/model/model.ckpt").to(device)
    # print(load_model)
    torchsummary.summary(load_model, input_size=(3, 224, 224))

    # Save and load only the model parameters (recommended).
    torch.save(resnet34.state_dict(), "./logs/model/model_params.ckpt")
    load_model = resnet34.load_state_dict(torch.load("./logs/model/model_params.ckpt"))


def main():
    torch.manual_seed(11)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # regression_torch_style_dataset()
    # regression_torch_style()
    # get_CIFAR10_dataset()
    # pretrained_model()
    # save_load_model()
    # get_MNIST_dataset()
    get_VocDetection()
    # get_CoCoDetection()


if __name__ == '__main__':
    main()
