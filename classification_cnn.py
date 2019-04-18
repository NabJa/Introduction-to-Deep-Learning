"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
import math
import numpy as np


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        pad = int((kernel_size - 1)/2)

        self.conv = nn.Conv2d(
                in_channels=channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride_conv,
                padding=pad
            )

        torch.nn.init.xavier_uniform_(self.conv.weight, weight_scale)

        #self.batch2d = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=kernel_size,
                stride=stride_conv,
                padding=pad
            )

        #self.batch2d1 = nn.BatchNorm2d(num_filters*2)

        # input volume: height * width * num_filters
        # output volume: (height-pool/stride_pool) * (width-pool/stride_pool) * num_filters
        self.max2pool = nn.MaxPool2d(
                kernel_size=pool,
                stride=stride_pool
            )

        self.fc1 = nn.Linear(
                #in_features=32 * 2 * 2,
                in_features=64 * 16 * 16,
                out_features=hidden_dim)

        self.drop = nn.Dropout2d(
                p=dropout
            )

        self.fc2 = nn.Linear(
                in_features=hidden_dim,
                out_features=num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        conv - relu - 2x2 max pool - fc - dropout - relu - fc

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max2pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)mv
        x = self.drop(x)
        x = F.relu(x)
        x = self.fc2(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        pad = int((kernel_size - 1) / 2)

        # input = 3 * 32 * 32
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride_conv,
            padding=pad
        )

        # conv = 32 * 32 * 32
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            stride=stride_conv,
            padding=pad
        )

        #conv2 = 64 * 32 * 32
        self.max2pool = nn.MaxPool2d(
            kernel_size=pool,
            stride=stride_pool
        )

        #max2pool = 64 * 16 * 16
        self.conv3 = nn.Conv2d(
            in_channels=num_filters * 2,
            out_channels=num_filters * 4,
            kernel_size=kernel_size,
            stride=stride_conv,
            padding=pad
        )

        #conv3 = 128 * 16 * 16
        self.conv4 = nn.Conv2d(
            in_channels=num_filters * 4,
            out_channels=num_filters * 8,
            kernel_size=kernel_size,
            stride=stride_conv,
            padding=pad
        )

        #conv3 = 256 * 16 * 16
        self.max2pool2 = nn.MaxPool2d(
            kernel_size=pool,
            stride=stride_pool
        )

        #max2pool2 = 256 * 8 * 8
        self.fc1 = nn.Linear(
            in_features=256 * 8 * 8,
            out_features=5000)

        self.drop = nn.Dropout2d(
            p=dropout
        )

        self.fc2 = nn.Linear(
            in_features=5000,
            out_features=100)

        self.fc3 = nn.Linear(
            in_features=100,
            out_features=num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        conv - relu - 2x2 max pool - fc - dropout - relu - fc

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max2pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2pool2(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
