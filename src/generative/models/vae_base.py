

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Vae(object):
  
    def __init__(self, latent_variable_dim, image_size):
        self.latent_variable_dim = latent_variable_dim
        self.image_size = image_size
        self.batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
  
    def encoder(self, images, is_training):
        # Must be overridden in implementation classes
        raise NotImplementedError
      
    def decoder(self, latent_var, is_training):
        # Must be overridden in implementation classes
        raise NotImplementedError

    def get_image_size(self):
        return self.image_size
        