#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torchvision import transforms
from sklearn.datasets import load_digits

import utils

SKLEARN_DIGITS_TRAIN_SIZE = 1247
SKLEARN_DIGITS_VAL_SIZE = 550
class CNN(nn.Module):

    def __init__(self, dropout_prob):
        """
        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a CNN module has convolution,
        max pooling, activation, linear, and other types of layers. For an 
        idea of how to us pytorch for this have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        super(CNN, self).__init__()

        # Implement me!

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=1, padding="same")
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=2)

        # initialize second set of CONV => RELU => POOL layers
        # self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding="valid")
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=2)

        # initialize first (and only) set of FC => RELU layers
        #print(self.conv2.weight.shape)
        # torch.Size([10, 3, 5, 5])
        self.fc1 = Linear(in_features=16 * 3 * 3, out_features=600)
        self.relu3 = ReLU()

        self.dropout_prob = nn.Dropout(p=0.3)

        self.fc2 = Linear(in_features=600, out_features=120)
        self.relu4 = ReLU()

        self.fc3 = Linear(in_features=120, out_features=10)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        """
        x (batch_size x n_channels x height x width): a batch of training 
        examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. This method needs 
        to perform all the computation needed to compute the output from x. 
        This will include using various hidden layers, pointwise nonlinear 
        functions, and dropout. Don't forget to use logsoftmax function before 
        the return

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        #print(x.shape)
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.dropout_prob(x)

        x = self.fc2(x)
        x = self.relu4(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    #X = X.view(X.size(0), -1)
    #print(X.shape)
    optimizer.zero_grad()
    output = model(X)
    #output = model(X.view(-1, 10))
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def plot_feature_maps(model, train_dataset):
    model.conv1.register_forward_hook(get_activation('conv1'))

    data, _ = train_dataset[4]
    data.unsqueeze_(0)
    output = model(data)

    plt.imshow(data.reshape(28, -1))
    plt.savefig('original_image.pdf')

    k = 0
    act = activation['conv1'].squeeze()
    fig, ax = plt.subplots(2, 4, figsize=(12, 8))

    for i in range(act.size(0) // 3):
        for j in range(act.size(0) // 2):
            ax[i, j].imshow(act[k].detach().cpu().numpy())
            k += 1
            plt.savefig('activation_maps.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.8)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')

    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()

    digits_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    X, y = load_digits(return_X_y=True)
    X = X.reshape((len(X), 8, 8))
    dev_y = y[:-SKLEARN_DIGITS_VAL_SIZE] #y_train
    test_y = y[-SKLEARN_DIGITS_VAL_SIZE:] #y_val
    dev_X = X[:-SKLEARN_DIGITS_VAL_SIZE]
    test_X = X[-SKLEARN_DIGITS_VAL_SIZE:]

    #train
    #val

    digits_train_dataset = utils.NumpyDataset(dev_X, dev_y, transform=digits_transform)
    digits_val_dataset = utils.NumpyDataset(test_X, test_y, transform=digits_transform)
    digits_train_dataloader = torch.utils.data.DataLoader(digits_train_dataset, batch_size=64, shuffle=True)
    digits_val_dataloader = torch.utils.data.DataLoader(digits_val_dataset, batch_size=64, shuffle=True)

    #dataloaders = dict(train=digits_train_dataloader, val=digits_val_dataloader)
    train_dataloader = dict(train=digits_train_dataloader, val=digits_val_dataloader)
    #print(train_dataloader['train'])
    #dataset = utils.ClassificationDataset(data)
    dataset = dict(train=digits_train_dataloader, val=digits_val_dataloader)
    train_dataloader1 = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    # print("****")
    # print(train_dataloader1)
    # print(train_dataloader['train'])

    # dev_X, dev_y = dataset.dev_X, dataset.dev_y
    # test_X, test_y = dataset.test_X, dataset.test_y


    # sample = enumerate(train_dataloader)
    # batch_id, (sample_data, sample_targets) = next(sample)
    # print(sample_data.shape)

    # initialize the model
    model = CNN(opt.dropout)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        #for X_batch, y_batch in train_dataloader:
        for X_batch, y_batch in digits_train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))

    plot_feature_maps(model, dataset)


if __name__ == '__main__':
    main()
