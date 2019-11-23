import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################

  scaler = StandardScaler()
  for file in glob.glob(f"{dir_name}/*/*/*.jpg"):
    img = Image.open(file).convert("L")
    img = np.array(img).reshape(-1, 1) / 255
    scaler.partial_fit(img)
  
  mean = scaler.mean_
  std = np.sqrt(scaler.var_)

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
