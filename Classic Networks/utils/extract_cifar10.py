import numpy as np
import os

import tensorflow as tf

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000
TEST_NUM = 10000


def extract_data(filenames):
    # to verify whether the file exists
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # read data
    labels = None
    images = None
    # whether reading the first file.
    flag = True

    for f in filenames:
        bytestream = open(f, 'rb')
        # read data
        buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))
        # convert data flow to numpy array
        data = np.frombuffer(buf, dtype=np.uint8)
        # change data format
        data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)  # data的每个图像，label在前，image在后
        # divide data
        labels_images = np.hsplit(data, [LABEL_SIZE])

        label = labels_images[0].reshape(TRAIN_NUM)
        image = labels_images[1].reshape(TRAIN_NUM, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        if flag:
            labels = label
            images = image
            flag = False
        else:
            # amalgamate array
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))
        pass

    # data processing for convenient computation
    images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    return labels, images


def extract_train_data(files_dir):
    """ get train data """
    filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    return extract_data(filenames)


def extract_test_data(files_dir):
    """ get test data """
    filenames = [os.path.join(files_dir, 'test_batch.bin')]
    return extract_data(filenames)


def dense_to_one_hot(labels_dense, num_classes):
    """ convert dense labels to one-hot labels """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class Cifar10DataSet(object):
    """ docstring for Cifar10 DataSet """

    def __init__(self, data_dir):
        super(Cifar10DataSet, self).__init__()
        self.train_labels, self.train_images = extract_train_data(
            os.path.join(data_dir, 'cifar-10-batches-bin'))
        self.test_labels, self.test_images = extract_test_data(os.path.join(data_dir, 'cifar-10-batches-bin'))

        # print(self.train_labels.size)

        self.train_labels = dense_to_one_hot(self.train_labels, NUM_CLASSES)
        self.test_labels = dense_to_one_hot(self.test_labels, NUM_CLASSES)

        # number of epoch completed
        self.epochs_completed = 0
        # current epoch in process
        self.index_in_epoch = 0

    def next_train_batch(self, batch_size):
        # initial position
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # print "self.index_in_epoch: ",self.index_in_epoch
        # complete one epoch
        if self.index_in_epoch > TRAIN_NUMS:
            self.epochs_completed += 1
            # print "self.epochs_completed: ",self.epochs_completed
            # shuffle the data
            perm = np.arange(TRAIN_NUMS)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= TRAIN_NUMS

        end = self.index_in_epoch
        # print "start,end: ",start,end

        return self.train_images[start:end], self.train_labels[start:end]

    def test_data(self):
        return self.test_images, self.test_labels


# def main():
#     # train_labels, train_images = extract_train_data('./data/cifar-10-batches-bin')
#     # print(train_images.shape)
#     cc = Cifar10DataSet('../data/')
#     cc.next_train_batch(100)
#
#
# if __name__ == '__main__':
#     main()
