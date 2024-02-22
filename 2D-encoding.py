
import sys
sys.path.insert(1,"..")
import matplotlib.pyplot as plt
import random

import os
from torchvision import transforms
import numpy as np
from torchvision import transforms
import albumentations as A
from imgaug import augmenters as iaaa
from PIL import Image
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from pyts.approximation import PiecewiseAggregateApproximation as PAA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import torch

from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot, MarkovTransitionField, GramianAngularField
from skimage.transform import resize
import cv2


# Load the dataset
data = np.load('specify the data path')

# Split the data into training and validation sets
X_train, X_valid = train_test_split(data, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Define the transformation methods
def transform_gadf(data):
     def __init__(self, image_size=32, overlapping=False, scale=-1):
        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        # Check input data
        X = check_array(X)
        # Shape parameters
        n_samples, n_features = X.shape
        # Check parameters
        paa = PAA(output_size=self.image_size, overlapping=self.overlapping)
        X_paa = paa.fit_transform(X)
        n_features_new = X_paa.shape[1]
        scaler = MinMaxScaler(feature_range=(self.scale, 1))
        X_scaled = scaler.fit_transform(X_paa.T).T
        X_sin = np.sqrt(np.clip(1 - X_scaled**2, 0, 1))
        X_scaled_sin = np.hstack([X_scaled, X_sin])
        X_scaled_sin_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 True)
        X_sin_scaled_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 False)
        return X_sin_scaled_outer - X_scaled_sin_outer
    def _outer_stacked(self, arr, size, first=True):
        if first:
            return np.outer(arr[:size], arr[size:])
        else:
            return np.outer(arr[size:], arr[:size])

def transform_gasf(data):
    gasf = GramianAngularField()

def transform_mtf(data):
   mtf = MarkovTransitionField()

def transform_rp(data):
    rp= RecurrencePlot()


# Create the folder if it doesn't exist
folder_path = '/mnt/DATA/AZZA/3_pixel_object_2D/transformation/reunion/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Loop over the 5 splits
for i in range(5):
    # Load the data for the current split
    valid_gadf = np.load(os.path.join(folder_path, f'GADF/valid_X_dev_{i}.npy'))
    valid_gasf = np.load(os.path.join(folder_path, f'GASF/valid_X_dev_{i}.npy'))
    valid_mtf = np.load(os.path.join(folder_path, f'MTF/valid_X_dev_{i}.npy'))
    valid_rp = np.load(os.path.join(folder_path, f'RP/valid_X_dev_{i}.npy'))

    # Transform and save GADF
    gadf_resized = transform_gadf(valid_gadf)
    np.save(os.path.join(folder_path, f'GADF/valid_X_dev_{i}.npy'), gadf_resized)

    # Transform and save GASF
    gasf_resized = transform_gasf(valid_gasf)
    np.save(os.path.join(folder_path, f'GASF/valid_X_dev_{i}.npy'), gasf_resized)

    # Transform and save MTF
    mtf_resized = transform_mtf(valid_mtf)
    np.save(os.path.join(folder_path, f'MTF/valid_X_dev_{i}.npy'), mtf_resized)

    # Transform and save RP
    rp_resized = transform_rp(valid_rp)
    np.save(os.path.join(folder_path, f'RP/valid_X_dev_{i}.npy'), rp_resized)

