#!/usr/bin/python3

import numpy as np
import math


def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = floor(k / 2)
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors with values populated from evaluating the 1D Gaussian PDF at each
    corrdinate.
  """

  ############################
  ### TODO: YOUR CODE HERE ###

  k = 4 * cutoff_frequency + 1
  mean = k // 2
  std = cutoff_frequency
  # I am assuming the range is [mean - k/2, mean + k/2]
  x_range = np.arange(start = mean - k//2, stop = mean + k//2 + 1)
  # Apply 1D gaussian formula
  gkernel_1d = 1/(np.sqrt(2*np.pi)*std) * np.exp( (-1/(2*std**2) * (x_range-mean)**2) )
  # outer product to get 2d
  kernel = np.outer(gkernel_1d, gkernel_1d) 
  # normalize value to get values that sum to 1
  kernel = (1 / kernel.sum()) * kernel

  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filter_row_size = filter.shape[0]
  filter_col_size = filter.shape[1]
  padded_input_image = np.pad(image, ((filter_row_size//2, filter_row_size//2), (filter_col_size//2, filter_col_size//2), (0,0)), "constant")
  filtered_image = np.empty(image.shape)
  for k in range(padded_input_image.shape[2]):
    for j in range(padded_input_image.shape[1]):
      if padded_input_image.shape[1] - j < filter_col_size:
        break
      for i in range(padded_input_image.shape[0]):
        if padded_input_image.shape[0] - i < filter_row_size:
          break
        filtered_image[i, j, k] = np.multiply(filter, padded_input_image[i:i+filter_row_size, j:j+filter_col_size, k]).sum()
        

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  low_frequencies = my_imfilter(image1, filter)
  # low_frequencies = (1/low_frequencies.sum()) * low_frequencies
  high_frequencies = image2 - my_imfilter(image2, filter)
  #high_frequencies = (1/high_frequencies.sum()) * high_frequencies
  hybrid_image = (high_frequencies+low_frequencies)
  hybrid_image = np.clip(hybrid_image, 0, 1)
  #hybrid_image = hybrid_image / hybrid_image.max()


  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
