import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch

from generator import Generator


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples, file_name=None, show=False):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()


def main():
    # Load samples from generator, taken while training
    with open('train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)

    # View a sample from the final epoch, -1 indicates final epoch
    view_samples(-1, samples, "training_last_epoch.png")

    rows = 10  # split epochs into 10, so 100/10 = every 10 epochs
    cols = 6
    fig, axes = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)
    for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes):
        for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
            img = img.detach()
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.savefig("training_epochs.png")
    plt.show()
    plt.close()

    # Generator hyperparams
    z_size = 100
    g_output_size = 784
    g_hidden_size = 32
    G = Generator(z_size, g_hidden_size, g_output_size)
    G.load_state_dict(torch.load("G.pth"))
    G.eval()  # eval mode

    # # Sampling from the generator
    # randomly generated, new latent vectors
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()
    # generated samples
    rand_images = G(rand_z)

    # 0 indicates the first set of samples in the passed in list
    # and we only have one batch of samples, here
    view_samples(0, [rand_images], "resample.png")


if __name__ == '__main__':
    main()
