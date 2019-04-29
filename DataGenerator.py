#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:50:58 2017

@author: pivotalit
"""

import numpy as np

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 1024, dim_y = 1024,  batch_size = 8, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1: 
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(list_IDs_temp)

              yield (X, y)

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.zeros((self.batch_size, self.dim_x, self.dim_y,  3), dtype=np.float32)
      y = np.ndarray((self.batch_size, self.dim_x, self.dim_y, 1), dtype=np.float32)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          pth='/users/pivotalit/downloads/training_1024_npy/' + str(ID) + '.npy'
          X[i] = np.load(pth)
          X[i] /= 255.
          # Store class
          pth='/users/pivotalit/downloads/training_1024_npy/' + str(ID) + '_mask.npy'
          y[i] = np.load(pth)
          y[i] /= 255.
      return X, y