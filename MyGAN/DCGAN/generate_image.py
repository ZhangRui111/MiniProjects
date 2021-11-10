import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision.utils import save_image

from generator import Generator


# helper function for viewing a list of passed in sample images
def view_samples(samples, sample_size):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((sample_size, sample_size)), cmap='Greys_r')
    plt.show()


def save_samples(samples, sample_size, save_path):
    for i, img in enumerate(samples):
        img = img.detach()
        img.reshape((sample_size, sample_size))
        save_image(img, "{}/img_{:04d}.png".format(save_path, i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, help="epoch of loaded ckp")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    opt = parser.parse_args()
    print(opt)

    images_path = "logs/output"
    os.makedirs(images_path, exist_ok=True)

    # # Sampling from the generator
    # randomly generated, new latent vectors
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, opt.latent_dim))
    rand_z = torch.from_numpy(rand_z).float()

    G = Generator(opt)
    G.load_state_dict(torch.load("logs/ckps/G_{}.pth".format(opt.epoch)))
    G.eval()  # eval mode
    # generated samples
    rand_images = G(rand_z)
    # view_samples(rand_images, opt.img_size)
    save_samples(rand_images, opt.img_size, images_path)


if __name__ == '__main__':
    main()
