#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
from torch import cuda, no_grad

import torch.nn as nn
from torch.optim import Adam

from cnn_system import MyCNNSystem
# .........
from pathlib import Path
from data_loader import load_data
# ............

__docformat__ = 'reStructuredText'



def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    epochs = 100

    # Instantiate our DNN
    #..................
    # Define the CNN model and give it the model hyperparameters
    cnn_model = MyCNNSystem(
        cnn_channels_1=16,
        cnn_kernel_1=3,
        cnn_stride_1=1,
        cnn_padding_1=1,
        pooling_kernel_1=2,
        pooling_stride_1=2,
        cnn_channels_2=32,
        cnn_kernel_2=3,
        cnn_stride_2=1,
        cnn_padding_2=1,
        pooling_kernel_2=2,
        pooling_stride_2=2,
        classifier_input_features=51520,
        output_classes=1,
        dropout=0.25)
    #.................

    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(cnn_model.parameters(), lr=1e-3)

    # Instantiate the loss function as a class.
    loss_function = nn.BCELoss()

    # Init training, validation, and testing dataset.
    data_path = Path('../music_speech_dataset')
    batch_size = 4
    
    # ............
    split = "training"
    train_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last= True, num_workers=1)
    
    split = 'validation'
    valid_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last= False, num_workers=1)
    
    split = 'testing'
    test_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last= False, num_workers=1)
   

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        cnn_model.train()

        # For each batch of our dataset.
        for batch in train_loader:
            
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch

            y = y.view(-1, 1).float()
                   
            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)
            
            # Get the predictions .
            y_hat = cnn_model(x)

            # Calculate the loss .
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        cnn_model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in valid_loader:
                # Get the batch
                x_val, y_val = batch

                y_val = y_val.view(-1, 1).float()
                
                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # Get the predictions of the model.
                y_hat = cnn_model(x_val)

                # Calculate the loss.
                loss = loss_function(y_hat, y_val)

                # Append the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(cnn_model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if (patience_counter >= patience) or (epoch==epochs-1):
            
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                # Load best model 
                cnn_model.load_state_dict(best_model)
                cnn_model.eval()
                with no_grad():
                    for batch in test_loader:
                        x_test, y_test = batch 

                        y_test = y_test.view(-1, 1).float()

                        # Pass the data to the appropriate device.
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)

                        # make the prediction
                        y_hat = cnn_model(x_test)
                        
                        # Calculate the loss.
                        loss = loss_function(y_hat, y_test)

                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# EOF
# Observations: I have change the convolutional filters, kernel sizes, and pooling sizes.
# - For the convolutional filters, I have used 16, 32 and 64 filters respectively. 
# During the training, the model was able to learn the features of the data and the loss was decreasing. 
# However, the higher the number of filters, the longer the training time. Also, training loss and validation loss were decreasing.
# However the model was overfitting the data easier, as the validation loss was increasing after a certain epoch, or remain consistent.
# - For the kernel sizes, I have used 3x3, 5x5 and 1x1 filters for all the convolutional layers.
# The model was able to learn the features of the data and the loss was decreasing.
# However, similar to the convolutional filters, the higher the kernel size, the longer the training time. 
# Also, training loss and validation loss were decreasing, but fluctuating towards the end of the training, with the training being stopped due to early stopping.
# - For the pooling sizes, I have used 2x2, 3x3 and 1x1 filters for all the pooling layers.
# The model was able to learn the features of the data and the loss was decreasing.
# However, similar to the convolutional filters, the higher the pooling size, the higher the computational cost.

