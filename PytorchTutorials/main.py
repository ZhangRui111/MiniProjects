"""
Based on https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
"""
import numpy as np
import torch


def demo():
    # Data Generation
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2*x + 0.1*np.random.randn(100, 1)

    # # Generates train and validation sets by numpy
    # idx = np.arange(100)
    # np.random.shuffle(idx)
    # train_idx = idx[:80]
    # val_idx = idx[80:]
    # x_train, y_train = x[train_idx], y[train_idx]
    # x_val, y_val = x[val_idx], y[val_idx]
    # x_train_tensor = torch.from_numpy(x_train).float()
    # y_train_tensor = torch.from_numpy(y_train).float()
    # x_val_tensor = torch.from_numpy(x_val).float()
    # y_val_tensor = torch.from_numpy(y_val).float()
    #
    # from torch.utils.data import TensorDataset, DataLoader
    # train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    # train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    # val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # Generates train and validation sets by torch's random_split()
    from torch.utils.data import TensorDataset, DataLoader
    from torch.utils.data.dataset import random_split
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    train_dataset, val_dataset = random_split(dataset, [80, 20])
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    import torch.nn as nn
    net = nn.Sequential(nn.Linear(1, 1)).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = nn.MSELoss(reduction='mean')
    losses = []

    for i in range(100):
        for x_batch, y_batch in train_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            net.train()
            prediction = net(x_batch)
            loss = loss_func(prediction, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for val_x_batch, val_y_batch in val_data_loader:
                    val_x_batch = val_x_batch.to(device)
                    val_y_batch = val_y_batch.to(device)
                    net.eval()
                    val_pred = net(val_x_batch)
                    val_loss = loss_func(val_pred, val_y_batch)
                    losses.append(val_loss)

    print(net.state_dict())
    print(losses)


def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo()


if __name__ == '__main__':
    main()
