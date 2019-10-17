"""
Practice of "Understanding torch with an example: a step-by-step tutorial"
from https://towardsdatascience.com/understanding-torch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
"""
import numpy as np
import torch


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


def regression_numpy_style(x_train, y_train):
    """
    Solve a Regression problem in Numpy style.
    :param x_train:
    :param y_train:
    :return:
    """
    # Initializes parameters "a" and "b" randomly
    np.random.seed(11)
    a = np.random.randn(1)
    b = np.random.randn(1)
    print(a, b)

    lr = 1e-1  # Sets learning rate
    n_epochs = 1000  # Defines number of epochs

    for epoch in range(n_epochs):
        # Computes our model's predicted output
        y_hat = a + b * x_train
        # How wrong is our model -- error
        error = (y_train - y_hat)
        # Mean squared error (MSE)
        loss = (error ** 2).mean()

        # Computes gradients for both "a" and "b" parameters
        a_grad = -2 * error.mean()
        b_grad = -2 * (x_train * error).mean()

        # Updates parameters using gradients and the learning rate
        a = a - lr * a_grad
        b = b - lr * b_grad

    print(a, b)

    # # Sanity Check: do we get the same results as our gradient descent?
    # from sklearn.linear_model import LinearRegression
    # linr = LinearRegression()
    # linr.fit(x_train, y_train)
    # print(linr.intercept_, linr.coef_[0])


def load_data2torch_tensor(x_train, y_train):
    """
    Load numpy arrays as torch tensors, or turn torch tensors back into numpy arrays.
    :param x_train:
    :param y_train:
    :return:
    """
    # Our data was in Numpy arrays, but we need to transform them into torch's Tensors
    # and then we send them to the chosen device, CPU or GPU.
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    # # Here we can see the difference -- notice that .type() is more useful
    # # since it also tells us WHERE the tensor is (device)
    # print(type(x_train), type(x_train_tensor), x_train_tensor.type())

    # We can also turn tensors back into Numpy arrays. But .numpy() cannot handle
    # GPU tensors. You need to make them CPU tensors first using cpu().
    # CPU tensors
    # x_train_ndarray = x_train_tensor.numpy()
    # GPU tensors
    x_train_ndarray = x_train_tensor.cpu().numpy()

    return x_train_tensor, y_train_tensor


def regression_torch_style_naive(x_train, y_train):
    lr = 1e-1
    n_epoch = 1000

    x_train_tensor, y_train_tensor = load_data2torch_tensor(x_train, y_train)
    # Create trainable parameters.
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    for i in range(n_epoch):
        y_hat = a + b * x_train_tensor
        error = y_train_tensor - y_hat
        loss = (error ** 2).mean()

        # .backward() computes all gradients from the specified loss!
        loss.backward()
        # # We can inspect the actual values of the gradients by looking
        # # at the .grad attribute of a tensor.
        # print(a.grad)
        # print(b.grad)

        # .no_grad() allows us to perform regular Python operations on tensors,
        # independent of torch's computation graph.
        # This is related to torch’s ability to build a dynamic computation graph
        # from every Python operation that involves any gradient-computing tensor
        # or its dependencies.
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad

        # Gradients are accumulated. So, every time we use the gradients to update
        # the parameters, we need to zero the gradients afterwards.
        a.grad.zero_()
        b.grad.zero_()

    print(a, b)


def regression_torch_style(x_train, y_train):
    lr = 1e-1
    n_epoch = 1000

    x_train_tensor, y_train_tensor = load_data2torch_tensor(x_train, y_train)
    # Creat trainable parameters.
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD([a, b], lr=lr)

    for i in range(n_epoch):
        y_hat = a + b * x_train_tensor

        # error = y_train_tensor - y_hat
        # loss = (error ** 2).mean()
        loss = loss_fn(y_train_tensor, y_hat)

        # .backward() computes all gradients from the specified loss!
        loss.backward()

        # with torch.no_grad():
        #     a -= lr * a.grad
        #     b -= lr * b.grad
        optimizer.step()

        # a.grad.zero_()
        # b.grad.zero_()
        optimizer.zero_grad()

    print(a, b)


class ManualLinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        # So we can retrieve an iterator over all model's parameters by model's parameters()
        # Moreover, we can get the current values for all parameters using our model’s state_dict()
        self.a = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float, device=device))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float, device=device))

        # # Nested model
        # # Instead of our custom parameters, we use a Linear layer with single input and single output
        # self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.a + self.b * x
        # # Nested model
        # return self.linear(x)


def regression_torch_style_model(x_train, y_train):
    # Make sure sending our model to the same device where the data is.
    # model = ManualLinearRegression().to(device)
    # Alternatively, you can use a Sequential model
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)
    print(model.state_dict())
    # print(model.parameters())

    lr = 1e-1
    n_epochs = 1000

    x_train_tensor, y_train_tensor = load_data2torch_tensor(x_train, y_train)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # model.train() method does NOT perform a training step.
        # Its only purpose is to set the model to training mode.
        # Because some models may use mechanisms like Dropout, for instance,
        # which have distinct behaviors in training and evaluation phases.
        model.train()

        # y_hat = a + b * x_tensor
        y_hat = model(x_train_tensor)

        loss = loss_fn(y_train_tensor, y_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(model.state_dict())
    # print(model.parameters())


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
        y_hat = model(x)
        # Computes loss
        loss = loss_fn(y, y_hat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step


def regression_torch_style_final(x_train, y_train):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(device)
    print(model.state_dict())

    lr = 1e-1
    n_epochs = 1000
    x_train_tensor, y_train_tensor = load_data2torch_tensor(x_train, y_train)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Creates the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []

    # For each epoch.
    for epoch in range(n_epochs):
        # Performs one train step and returns the corresponding loss
        loss = train_step(x_train_tensor, y_train_tensor)
        losses.append(loss)

    # Checks model's parameters
    print(model.state_dict())


def main():
    torch.manual_seed(11)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, y_train, x_val, y_val = data_generation()

    # # Regression in Numpy style.
    # regression_numpy_style(x_train, y_train)

    # # regression in torch style.
    # regression_torch_style_naive(x_train, y_train)
    # regression_torch_style(x_train, y_train)
    regression_torch_style_model(x_train, y_train)
    # regression_torch_style_final(x_train, y_train)


if __name__ == '__main__':
    main()
