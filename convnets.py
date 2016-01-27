#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import gc
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from data_utils import MRIDataIterator

def build_cnn(input_var=None, numer_of_buckets=10):
    """number_of_buckets:   Is the number of histogram buckets we have created.
                            We treat these like layers for the convolution,
                            filling in the missing layers with 0s. We also throw
                            out slices that are probably from the same location
    """

    # Input layer, as usual:
    # (number of frames in cardiac cycle x number_of_buckets x image_width x image_height)
    # (30 x 10 x 64 x 64)
    network = lasagne.layers.InputLayer(shape=(30, numer_of_buckets, 64, 64),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.FlattenLayer(network, 2)
    #should now be (30 x (64*64*10))

    print("After flatter, dims: {}".format(network.output_shape))

    # need to get it to (1 x 30 x (64*64*10))
    network = lasagne.layers.ReshapeLayer(network, (-1, [0], [1]))

    print("After reshape, dims: {}".format(network.output_shape))

    network = lasagne.layers.LSTMLayer(
        network, 512, grad_clipping=100,
        nonlinearity=lasagne.nonlinearities.tanh)

    print("After lstm, dims: {}".format(network.output_shape))

    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer.
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    network = lasagne.layers.SliceLayer(network, -1, 1)

    print("After slice, dims: {}".format(network.output_shape))

    # A fully-connected layer of 1024 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 600-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=600,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=1):
    # Load the dataset
    print("Creating data iterator...")
    mriIter = MRIDataIterator("/Users/Breakend/Documents/datasets/sciencebowl2015/train", "/Users/Breakend/Documents/datasets/sciencebowl2015/train.csv")

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = theano.tensor.nnet.categorical_crossentropy(prediction, target_var).mean()
    # loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.

    #TODO: this should actually be CPRS

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        training_index = 1
        validation_index = mriIter.last_training_index + 1

        start_time = time.time()
        while mriIter.has_more_training_data(training_index):
            gc.collect()
            print("Training index %s" % training_index)
            try:
                inputs, systole, diastole = mriIter.retrieve_data_batch_by_layer_buckets(training_index)
            except:
                print("Skipping because failed to retrieve data")
                training_index += 1
                continue
            # systole, diastole = targets
            # import pdb;pdb.set_trace()
            print("Inputs shape: {}".format(inputs.shape))
            train_err += train_fn(inputs, systole)
            train_batches += 1
            training_index += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        while mriIter.has_more_validation_data(validation_index):
            gc.collect()
            print("Validation index %s" % validation_index)
            try:
                inputs, systole, diastole = mriIter.retrieve_data_batch_by_layer_buckets(validation_index)
            except:
                print("Skipping because failed to retrieve data")
                continue
            # systole, diastole = targets
            err, acc = val_fn(inputs, systole)
            val_err += err
            val_acc += acc
            val_batches += 1
            validation_index += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
