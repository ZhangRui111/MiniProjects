"""
Based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
"""
import argparse
import os
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
from torch.utils.data import DataLoader

from dataloader_xcad import XCADImageDataset
from generator import Generator
from discriminator import Discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=['xcad', 'mnist'], help="type of real images")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument("--ckp_interval", type=int, default=5, help="interval between model saving")
    opt = parser.parse_args()
    print(opt)

    images_path = "logs/images/{}".format(opt.data)
    os.makedirs(images_path, exist_ok=True)
    ckps_path = "logs/ckps/{}".format(opt.data)
    os.makedirs(ckps_path, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    os.makedirs("../data", exist_ok=True)
    if opt.data == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )
    elif opt.data == 'xcad':
        transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomRotation(180),  # add diversity
            transforms.RandomHorizontalFlip(0.5),  # add diversity
            transforms.RandomVerticalFlip(0.5),  # add diversity
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = XCADImageDataset(
            "../data",
            transform=transform
        )
    else:
        raise ValueError("invalid args --data")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.ones(imgs.shape[0], 1)
            fake = torch.zeros(imgs.shape[0], 1)
            if cuda:
                imgs = imgs.cuda()
                valid = valid.cuda()
                fake = fake.cuda()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.from_numpy(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).float()
            if cuda:
                z = z.cuda()

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(
                epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "{}/{}.png".format(images_path, batches_done), nrow=5, normalize=True)

        if (epoch + 1) % opt.ckp_interval == 0:
            torch.save(generator.state_dict(), "{}/G_{}.pth".format(ckps_path, epoch))
            torch.save(discriminator.state_dict(), "{}/D_{}.pth".format(ckps_path, epoch))

    torch.save(generator.state_dict(), "{}/G_last.pth".format(ckps_path))
    torch.save(discriminator.state_dict(), "{}/D_last.pth".format(ckps_path))


if __name__ == '__main__':
    main()
