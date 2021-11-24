import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

from discriminator import Discriminator
from generator import Generator
from vis import view_samples


def real_loss(criterion, preds, labels, smooth=False):
    # label smoothing
    if smooth:
        labels = labels * 0.9

    loss = criterion(preds.squeeze(), labels)
    return loss


def fake_loss(criterion, preds, labels):
    loss = criterion(preds.squeeze(), labels)
    return loss


def set_model_gradient(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def main():
    # # -------------------- Data --------------------
    num_workers = 8  # number of subprocesses to use for data loading
    batch_size = 64  # how many samples per batch to load
    transform = transforms.ToTensor()  # convert data to torch.FloatTensor
    train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    # # Obtain one batch of training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # images = images.numpy()
    # # Get one image from the batch for visualization
    # img = np.squeeze(images[0])
    # fig = plt.figure(figsize=(3, 3))
    # ax = fig.add_subplot(111)
    # ax.imshow(img, cmap='gray')
    # plt.show()

    # # -------------------- Discriminator and Generator --------------------
    # Discriminator hyperparams
    input_size = 784  # Size of input image to discriminator (28*28)
    d_output_size = 1  # Size of discriminator output (real or fake)
    d_hidden_size = 32  # Size of last hidden layer in the discriminator
    # Generator hyperparams
    z_size = 100  # Size of latent vector to give to generator
    g_output_size = 784  # Size of discriminator output (generated image)
    g_hidden_size = 32  # Size of first hidden layer in the generator
    # Instantiate discriminator and generator
    D = Discriminator(input_size, d_hidden_size, d_output_size)
    G = Generator(z_size, g_hidden_size, g_output_size)

    # # -------------------- Optimizers and Criterion --------------------
    # Training hyperparams
    num_epochs = 100
    print_every = 400
    lr = 0.002

    # Create optimizers for the discriminator and generator, respectively
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)
    losses = []  # keep track of generated "fake" samples

    criterion = nn.BCEWithLogitsLoss()

    # -------------------- Training --------------------
    D.train()
    G.train()

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    samples = []  # keep track of loss

    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)

            # Important rescaling step
            real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)

            # Generate fake images, used for both discriminator and generator
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)

            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================

            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            D_real = D(real_images)
            d_real_loss = real_loss(criterion, D_real, real_labels, smooth=True)

            # 2. Train with fake images

            # Compute the discriminator losses on fake images
            # -------------------------------------------------------
            # ATTENTION:
            # *.detach(), thus, generator is fixed when we optimize
            # the discriminator
            # -------------------------------------------------------
            D_fake = D(fake_images.detach())
            d_fake_loss = fake_loss(criterion, D_fake, fake_labels)

            # 3. Add up loss and perform backprop
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================

            g_optimizer.zero_grad()

            # Make the discriminator fixed when optimizing the generator
            set_model_gradient(D, False)

            # 1. Train with fake images and flipped labels

            # Compute the discriminator losses on fake images using flipped labels!
            G_D_fake = D(fake_images)
            g_loss = real_loss(criterion, G_D_fake, real_labels)  # use real loss to flip labels

            # 2. Perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Make the discriminator require_grad=True after optimizing the generator
            set_model_gradient(D, True)

            # =========================================
            #           Print some loss stats
            # =========================================
            if batch_i % print_every == 0:
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        # AFTER EACH EPOCH
        losses.append((d_loss.item(), g_loss.item()))

        # generate and save sample, fake images
        G.eval()  # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        view_samples(-1, samples, "last_sample.png")
        G.train()  # back to train mode

    # Save models and training generator samples
    torch.save(G.state_dict(), "G.pth")
    torch.save(D.state_dict(), "D.pth")
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # Plot the loss curve
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


if __name__ == '__main__':
    main()
