"""
GAN class definition
Basic structure copied from https://github.com/raspstephan/radar-gan/blob/master/models/GAN.py
"""

# Imports
import numpy as np
from .models import *
from .utils import *
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from PIL import Image
import os
import pickle


# GAN class
class GAN(object):
    """
    GAN class.
    """
    def __init__(self, verbose=0):
        """
        Initialize GAN object. Define attributes.

        Args:
            verbose: Verbosity level
        """
        # Initialize empty networks
        self.G = None    # Generator G(x)
        self.D = None    # Discriminator D(x)
        self.GD = None   # Combined D(G(x))

        # Initialize the data generators
        # Generators should be compatible to CBRAIN data generator
        self.train_generator=None
        self.valid_generator=None

        # Define basic properties of the network

        # Training information
        # Useful if training is not done in one go
        self.epoch_counter = 0
        self.train_history = OrderedDict({
            'train_discriminator_loss': [],
            'train_generator_loss': [],
            'valid_discriminator_loss': [],
            'valid_generator_loss': [],
        })
        self.verbose = verbose

    def create_generator(self, **kwargs):
        """
        Create the generator network.
        """
        self.G = create_generator(**kwargs)

    def create_discriminator(self, **kwargs):
        """
        Create the discriminator network
        """
        self.D = create_discriminator(**kwargs)

    def compile(self):
        """
        Compile the networks and create the combined network.
        """

        # Compile the individual models
        self.G.compile(optimizer=opt, loss='mse')  # Loss does not matter for G
        self.D.compile(optimizer=opt, loss=loss)

        # Create and compile the combined model
        self.D.trainable = False
        inp_latent = Input(shape=(self.latent_size,))
        self.GD = Model(inputs=inp_latent, outputs=self.D(self.G(inp_latent)))
        self.GD.compile(optimizer=opt, loss=loss)

    def load_data_generator(self, dataset, bs, **kwargs):
        """
        Load the training and validation data generators for the requested
        dataset.

        Args:
            dataset: Name of dataset
            bs: Batch size
        """
        self.train_generator=None
        self.valid_generator=None

    def train(self, epochs):
        """
        Training operation. Note that batch size is defined in the data
        generator.

        Args:
            epochs: Number of epochs to train
        """

        if self.verbose > 0: pbar = tqdm(total=epochs * n_batches)
        for e in range(self.epoch_counter, self.epoch_counter + epochs):
            dl, gl = [], []
            for b in range(n_batches):
                if self.verbose > 0: pbar.update(1)
                dl, gl = self.train_step(bs, dl, gl, train_D_separately,
                                         noise_shape, n_disc)
            self.epoch_counter += 1

            # End of epoch. Compute mean generator and discriminator loss
            self.train_history['train_discriminator_loss'].append(np.mean(dl))
            self.train_history['train_generator_loss'].append(np.mean(gl))
            fake = self.evaluate_test_losses(noise_shape, bs)

            # Save images
            if e % save_interval == 0:
                self.save_images(fake)

            # Update progressbar with latest losses
            if self.verbose > 0:
                pbar_dict = OrderedDict({k: v[-1] for k, v
                                         in self.train_history.items()})
                pbar.set_postfix(pbar_dict)
        if self.verbose > 0: pbar.close()

    def train_step(self, bs, dl, gl, train_D_separately, noise_shape,
                   n_disc):
        """One training step. May contain several discriminator steps."""

        # STEP 1: TRAIN DISCRIMINATOR
        self.D.trainable = True

        for i_disc in range(n_disc):
            # Get images
            real = None

            # Create fake images
            fake = self.G.predict_on_batch()

            # Concatenate real and fake images and train the discriminator
            if train_D_separately:
                # Train on real data first
                tmp = self.D.train_on_batch(
                    real, np.array(self.label('real') * bs)
                )
                # Then on fake data
                tmp += self.D.train_on_batch(
                    fake, np.array(self.label('fake') * bs)
                )
                dl.append(tmp / 2.)
            else:
                X_concat = np.concatenate([real, fake])
                y_concat = np.array(
                    self.label('real') * bs + self.label('fake') * bs
                )
                dl.append(self.D.train_on_batch(X_concat, y_concat))


        # STEP 2: TRAIN GENERATOR
        self.D.trainable = False
        gl.append(self.GD.train_on_batch())
        return dl, gl

    def evaluate_test_losses(self, noise_shape, bs):
        """Compute losses for test set and returns some fake images"""
        fake = self.G.predict()
        X_concat = np.concatenate([self.X_test, fake])
        y_concat = np.array(
            self.label('real') * self.n_test +
            self.label('fake') * self.n_test
        )
        self.train_history['test_discriminator_loss'].append(
            self.D.evaluate(X_concat, y_concat, batch_size=bs, verbose=0)
        )
        self.train_history['test_generator_loss'].append(
            self.GD.evaluate())
        return fake

    def label(self, s):
        """Little helper function to return labels"""
        assert s in ['real', 'fake'], 'Wrong string for label function.'
        return [1] if s == 'real' else [0]

    def save_images(self, fake):
        """Saves some fake images"""
        s = (self.img_dir + '/' + self.exp_id + '_' +
               'plot_epoch_{0:04d}_generated'.format(self.epoch_counter))
        if self.dataset == 'mnist':
            # From https://github.com/lukedeo/keras-acgan/blob/master/mnist_acgan.py
            img = (np.concatenate(
                [fake[i * 3:(i + 1) * 3, :, :, 0].reshape(-1, self.image_size)
                 for i in range(3)],
                axis=-1
            ) * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save(s + '.png')
        if self.dataset == 'radar':
            np.save(s + '.npy', fake[:9])

    def save_models(self):
        """
        Saves models and training history
        """
        s = self.model_dir + self.exp_id + '_'
        self.G.save(s + 'G.h5')
        self.D.save(s + 'D.h5')
        self.GD.save(s + 'GD.h5')
        # Save training history
        with open(s + 'history.pkl', 'wb') as f:
            pickle.dump(self.train_history, f)