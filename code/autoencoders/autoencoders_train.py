"""
Train an autoencoders with `CNTK <https://github.com/Microsoft/CNTK>`_.
"""
import pandas
import numpy as np
import sys
import os
import cntk as C
from cntk import load_model

from cntk_image_reader import create_map_file, create_reader
from cntk import user_function

from autoencoders_train_pink import PinkActivation


def create_model(num_channels, image_width, image_height, layer, rconv, rpool, pink, beta):
    """
    Create a model.
    
    @param      num_channels    number channels
    @param      image_width     image width
    @param      image_height    image height
    @param      layer			number of groups of hidden layers (1 or 2)
    @param      pink            use the pink activation
    @param      beta            beta parameter for the image gradient
    @return                     (input_var, output_var), model, (losses,)
    """
    input_dim = image_height * image_width * num_channels

    # Input variable and normalization
    input_var = C.ops.input_variable((num_channels, image_height, image_width), np.float32)
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), input_var, name="input_node")
    
    conv_size = (rconv, rconv)
    cMap    = num_channels
    pool_size = (rpool, rpool)
    pad = True
    
    img_name = os.path.join(os.path.dirname(__file__), "data", "pink_elephant.jpg")
    
    # Define the auto encoder model
    if layer == 2:
        conv1   = C.layers.Convolution2D  (conv_size, num_filters=cMap, pad=pad, activation=C.ops.relu)(scaled_input)
        pool1   = C.layers.MaxPooling   (pool_size, pool_size, name="ae_node")(conv1)
        if pink:
            lay = user_function(PinkActivation(pool1, img_name, beta))
        else:
            lay == pool1
        unpool1 = C.layers.MaxUnpooling (pool_size, pool_size)(lay, conv1)
        deconv1       = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node_int")(unpool1)
        z = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node")(deconv1)
    elif layer == 0.5:
        conv1   = C.layers.Convolution2D (conv_size, num_filters=num_channels, pad=pad, 
                                                                        activation=C.ops.relu, name="ae_node")(scaled_input)
        if pink:
            lay = user_function(PinkActivation(conv1, img_name, beta))
        else:
            lay = conv1
        z = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node")(lay)        
    elif layer == 0.25:
        conv1   = C.layers.Convolution2D (conv_size, num_filters=num_channels, pad=pad, 
                                                                        activation=C.ops.relu, name="ae_node")(scaled_input)
        if pink:
            lay = user_function(PinkActivation(conv1, img_name, beta))
        else:
            lay = conv1
        z       = lay
    elif layer == 1:
        conv1   = C.layers.Convolution2D  (conv_size, num_filters=cMap, pad=pad, activation=C.ops.relu)(scaled_input)
        pool1   = C.layers.MaxPooling   (pool_size, pool_size, name="ae_node")(conv1)
        if pink:
            lay = user_function(PinkActivation(pool1, img_name, beta))
        else:
            lay = pool1
        unpool1 = C.layers.MaxUnpooling (pool_size, pool_size)(lay, conv1)
        z       = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node")(unpool1)        
    elif layer == "1d":
        conv1   = C.layers.Convolution2D  (conv_size, num_filters=cMap, pad=pad, activation=C.ops.relu)(scaled_input)
        pool1   = C.layers.MaxPooling   (pool_size, pool_size, name="ae_node")(conv1)
        if pink:
            lay = user_function(PinkActivation(pool1, img_name, beta))
        else:
            lay = pool1
        unpool1 = C.layers.MaxUnpooling (pool_size, pool_size)(lay, conv1)
        dense = C.layers.Dense((unpool1.shape), activation=C.relu,
                                            input_rank=None, name="dense")(unpool1)
        z = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node")(dense)
    elif layer == "0.5d":
        conv1   = C.layers.Convolution2D (conv_size, num_filters=num_channels, pad=pad, 
                                                                        activation=C.ops.relu, name="ae_node")(scaled_input)
        if pink:
            lay = user_function(PinkActivation(conv1, img_name, beta))
        else:
            lay = conv1
        dense = C.layers.Dense(lay.shape, activation=C.relu, name="dense")(lay)
        z = C.layers.ConvolutionTranspose2D(conv_size, num_filters=num_channels, pad=pad, bias=False, 
                                        init=C.glorot_uniform(0.001), name="output_node")(dense)        
    else:
        raise ValueError("Not implemented '{0}'".format(layer))

    # define rmse loss function (should be 'err = C.ops.minus(deconv1, scaled_input)')
    # input_dim = (image_height+4) * (image_width+4) * num_channels
    f2        = C.ops.element_times(C.ops.constant(0.0090625), input_var)
    minf   = C.ops.minus(z, f2)
    err       = C.ops.reshape(minf, (input_dim))
    sq_err    = C.ops.element_times(err, err)
    mse       = C.ops.reduce_mean(sq_err)
    rmse_loss = C.ops.sqrt(mse)
    rmse_eval = C.ops.sqrt(mse)
    return (input_var, input_var), z, (rmse_eval, rmse_loss)


def train_model(folder, inout, model, losses, 
                channels, width, height, 
                epoch_size=2000, minibatch_size=512, max_epochs=100,
                suffix="", lr=0.00015, deflearner="sgd"):
    """
    Train a model.
    
    :param folder: image folder (color)
    :param input: tuple (input_var, output_var)
    :param model: model to train
    :param losses: losses function (might be many)
    :param width: zoom before
    :param height: zoom before
    :param suffix: to distinguish between training
    """
    final = os.path.split(folder)[-1]
    map_file = os.path.join(os.path.dirname(__file__), "map_file_{0}.txt".format(final))
    if not os.path.exists(map_file):
        create_map_file(map_file, folder, "jpg", class_mapping=None, include_unknown=True)
    
    reader_train = create_reader(map_file, channels=channels, width=width, height=height)

    # Set learning parameters
    lr_schedule = C.learning_rate_schedule([lr], C.learners.UnitType.sample, epoch_size)
    mm_schedule = C.learners.momentum_as_time_constant_schedule([600], epoch_size)

    # Instantiate the trainer object to drive the model training
    if deflearner == "sgd":
        clearn =C.learners.momentum_sgd
    else:
        clearn = C.learners.adam
    learner = clearn(model.parameters, lr_schedule, mm_schedule, unit_gain=True)
    progress_printer = C.logging.ProgressPrinter(tag='Training')
    trainer = C.Trainer(model, losses, learner, progress_printer) 
    
    # define mapping from reader streams to network inputs
    
    input_var, output_var = inout
    input_map = { input_var : reader_train.streams.features }
    if input_var != output_var:
        input_map = { output_var : reader_train.streams.label }

    C.logging.log_number_of_parameters(model)
    print()
    
    if not os.path.exists(suffix):
        os.mkdir(suffix)

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), 
                                input_map=input_map)
            # update model with it
            trainer.train_minibatch(data)
            # count samples processed so far
            sample_count += data[input_var].num_samples                     

        trainer.summarize_training_progress()
        model.save(os.path.join("models", suffix, "ae_{}.model".format(epoch)))


if __name__=='__main__':
    this = os.path.abspath(os.path.dirname(__file__))
    folder = os.path.join(this, "101_ObjectCategories")
    channels = 3
    for layer, width, height, poolconv in [
                    #("1d", 50, 40, 3), 
                    #("1d", 64, 64, 3), 
                    #("0.5d", 64, 64, 3), 
                    (0.5, 64, 64, 3), 
                    (0.5, 64, 64, 5), 
                    (0.5, 192, 192, 5), 
                    (0.5, 192, 192, 3), 
                    (1, 100, 80, 5), 
                    (1, 100, 80, 3), 
                    (1, 64, 64, 3), 
                    (1, 200, 160, 5), 
                    (1, 200, 160, 3), 
                    (1, 192, 192, 5), 
                    (1, 192, 192, 3), 
                    (2, 100, 80, 5), 
                    (2, 100, 80, 3), 
                    (0.25, 64, 64, 5), 
                    ]:
        for pink in [True, False]:
            if pink:
                betas = [0.00001, 0.000001]
                lrs = [0.00002, 0.00001]
            else:
                betas = [0.00001]
                lr = [0.00002, 0.00001, 0.0001]
            for beta in betas:
                for lr in lrs:
                    for defle in ['adam', 'sgd']:
                        inout, model, losses = create_model(channels, width, height, layer, 
                                                            poolconv, poolconv, pink=pink, beta=beta)
                        suffix = "h{}_{}x{}_{}{}_lr{}_b{}_def{}".format(layer, width, height, poolconv, 
                                                                                     "_pink" if pink else "",
                                                                                     lr, beta, defle)
                        print("------------------------------------------")
                        print("suffix={0}".format(suffix))
                        print("------------------------------------------")
                        train_model(folder, inout, model, losses, channels, width, height,
                                          suffix=suffix, max_epochs=100, lr=0.0001, deflearner=defle)
