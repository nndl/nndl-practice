# python: 2.7
# encoding: utf-8

import numpy as np


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""

        # 请补全此处代码
        pass

    def train(self, data):
        """Train model using data."""

        # 请补全此处代码
        pass

    def sample(self):
        """Sample from trained model."""

        # 请补全此处代码
        pass


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print mnist.shape

    # construct rbm model
    rbm = RBM(2, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    s = rbm.sample()
