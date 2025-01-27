#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple

from torch import Tensor, cuda, rand
from torch.nn import Module, Conv2d, MaxPool2d, \
    BatchNorm2d, ReLU, Linear, Sequential, Dropout2d, Sigmoid

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['MyCNNSystem']


class MyCNNSystem(Module):

    def __init__(self,
                 cnn_channels_1: int,
                 cnn_kernel_1: Union[Tuple[int], int],
                 cnn_stride_1: Union[Tuple[int], int],
                 cnn_padding_1: Union[Tuple[int], int],
                 pooling_kernel_1: Union[Tuple[int], int],
                 pooling_stride_1: Union[Tuple[int], int],
                 cnn_channels_2: int,
                 cnn_kernel_2: Union[Tuple[int], int],
                 cnn_stride_2: Union[Tuple[int], int],
                 cnn_padding_2: Union[Tuple[int], int],
                 pooling_kernel_2: Union[Tuple[int], int],
                 pooling_stride_2: Union[Tuple[int], int],
                 classifier_input_features: int,
                 output_classes: int,
                 dropout: float) -> None:
        """MyCNNSystem, using two CNN layers, followed by a ReLU, a batch norm,\
        and a max-pooling process.

        :param cnn_channels_1: Output channels of first CNN.
        :type cnn_channels_1: int
        :param cnn_kernel_1: Kernel shape of first CNN.
        :type cnn_kernel_1: int|Tuple[int, int]
        :param cnn_stride_1: Strides of first CNN.
        :type cnn_stride_1: int|Tuple[int, int]
        :param cnn_padding_1: Padding of first CNN.
        :type cnn_padding_1: int|Tuple[int, int]
        :param pooling_kernel_1: Kernel shape of first pooling.
        :type pooling_kernel_1: int|Tuple[int, int]
        :param pooling_stride_1: Strides of first pooling.
        :type pooling_stride_1: int|Tuple[int, int]
        :param cnn_channels_out_2: Output channels of second CNN.
        :type cnn_channels_out_2: int
        :param cnn_kernel_2: Kernel shape of second CNN.
        :type cnn_kernel_2: int|Tuple[int, int]
        :param cnn_stride_2: Strides of second CNN.
        :type cnn_stride_2: int|Tuple[int, int]
        :param cnn_padding_2: Padding of second CNN.
        :type cnn_padding_2: int|Tuple[int, int]
        :param pooling_kernel_2: Kernel shape of second pooling.
        :type pooling_kernel_2: int|Tuple[int, int]
        :param pooling_stride_2: Strides of second pooling.
        :type pooling_stride_2: int|Tuple[int, int]
        :param classifier_input_features: Input features to the\
                                          classifier.
        :type classifier_input_features: int
        :param dropout: Dropout to use.
        :type dropout: float
        :param output_classes: Output classes.
        :type output_classes: int
        """
        super().__init__()
        
        self.block_1 = Sequential(
            Conv2d(in_channels=1, out_channels=cnn_channels_1,
                   kernel_size=cnn_kernel_1, stride=cnn_stride_1,
                   padding=cnn_padding_1),
            ReLU(),
            BatchNorm2d(num_features=cnn_channels_1),
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1),
            Dropout2d(p=dropout)
        )
        self.block_2 = Sequential(
            Conv2d(in_channels=cnn_channels_1, out_channels=cnn_channels_2,
                   kernel_size=cnn_kernel_2, stride=cnn_stride_2,
                   padding=cnn_padding_2),
            ReLU(),
            BatchNorm2d(num_features=cnn_channels_2),
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2)
        )
        self.classifier = Sequential(
            Linear(in_features=classifier_input_features, out_features=1),
            Sigmoid()  # Apply sigmoid activation to ensure output is between 0 and 1
        )


    def forward(self,
                x: Tensor)\
            -> Tensor:
        """Forward pass.

        :param x: Input features\
                  (shape either `batch x time x features` or\
                  `batch x channels x time x features`).
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = x if x.ndimension() == 4 else x.unsqueeze(1)
        
        # apply block_1 to h
        h = self.block_1(h)      
        
        # apply block_2 to h
        h = self.block_2(h)       
        
        # apply permute and reshaping
        h = h.permute(0, 2, 1, 3).contiguous().view(x.size()[0], -1)
        
        return self.classifier(h)


def main():
    
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    # Instantiate our CNN
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
    
    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)
    
    batch_size = 8
    d_time = 646
    d_feature = 40
    x = rand(batch_size,d_time,d_feature)
    y = rand(batch_size,1)
    
    # Give them to the appropriate device.
    x = x.to(device)
    y = y.to(device)

    # Get the predictions .
    y_hat = cnn_model(x)

    print(y_hat.shape)
    
    
    
if __name__ == '__main__':
    main()

# EOF
