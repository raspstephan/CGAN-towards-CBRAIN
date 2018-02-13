"""
Contains the actual model definitions.
Contains parts from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
and
https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py
"""

from keras.models import Model
from keras.layers import *
import keras.backend as K
assert K.image_data_format() == 'channels_first', 'Require channels_first!'


# Define blocks
def conv_block(x, filters, bn=True):
    """
    Convolution block: Conv2D -> Batch norm -> LeakyReLU

    Args:
        x: Input tensor [samples, channels/filters, height, width]
        filters: Number of output filters
        bn: Use batch norm

    Returns:
        x: Output tensor [samples, filters, height, width]
    """
    x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
    if bn:
        x = BatchNormalization(axis=1)(x)   # Channel axis
    x = LeakyReLU(0.2)(x)
    return x


def upsample_block(x, x_down, filters, bn=True, dr=0.5):
    """
    Upsampling block for Unet

    Args:
        x:
        filters:
        bn:
        dr:

    Returns:
        x:
    """
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=3)(x)
    if bn:
        x = BatchNormalization(axis=1)(x)
    if dr > 0:
        x = Dropout(dr)
    return concatenate(axis=1)([x, x_down])   # Concatenate along channel axis


# Define networks
def create_Unet_generator(img_dim, n_down, final_activation='tanh'):
    """
    Create the generator model: Unet
    For now only 3 channels to 3 channels.
    """

    # Define input
    inp = Input(shape=img_dim)
    x = inp

    # Go down
    down_list = []
    filter_list = []
    for n, n_filters in enumerate(filter_list):
        x = conv_block(x, n_filters)
        down_list.append(x)

    # Go up
    for x_down, n_filters in zip(down_list[::-1], filter_list[::-1]):
        x = upsample_block(x, x_down, n_filters)

    # Final activation
    outp = Activation(final_activation)(x)

    # Create and return model
    return Model(inputs=inp, outputs=outp)


def create_PixelGAN_discriminator(img_dim, n_filters_first=64, n_layers=3):
    """
    Create the discriminator model: PixelGAN

    Args:
        img_dim: Tuple of image dimensions [channels, height, width]
        n_filters_first: Number of convolution filters in the first layer
        n_layers: number of convolution layers including first

    Returns:
        model: Keras model
    """

    # Get input tensor
    inp = Input(shape=img_dim)

    # First convolution layer
    x = conv_block(inp, n_filters_first, bn=False)   # False in pytorch implentation

    # Loop over next layers
    n_filters = n_filters_first
    for n in range(1, n_layers):
        x = conv_block(x, n_filters)
        n_filters = min(2**n, 8) * n_filters_first

    # Final layer to scalar
    outp = Dense(1, activation='sigmoid')(Flatten()(x))

    # Create and return model
    return Model(inputs=inp, outputs=outp)
