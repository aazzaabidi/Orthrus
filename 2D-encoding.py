import os
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyts.approximation import PiecewiseAggregateApproximation as PAA
from pyts.image import RecurrencePlot, MarkovTransitionField, GramianAngularField
import sys
sys.path.insert(1,"..")


class GADF(BaseEstimator, TransformerMixin):
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
        return X_sin_scaled_outer - X_sin_scaled_outer

    def _outer_stacked(self, arr, size, first=True):
        if first:
            return np.outer(arr[:size], arr[size:])
        else:
            return np.outer(arr[size:], arr[:size])

def load_data_for_split(base_directory, split):
    return np.load(os.path.join(base_directory, f'splits/reunion/train_X_pxl_{split}.npy'))

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def transform_and_save_data(transformer, data, transformer_name, split, base_directory):
    transformed_data = transformer.transform(data)
    resized_data = list(map(lambda x: cv2.resize(x, (32, 32)).reshape(-1), transformed_data))
    output_path = os.path.join(base_directory, f'transformation/reunion/{transformer_name}/train_X_pxl_{split}.npy')
    create_directory(os.path.dirname(output_path))
    np.save(output_path, np.array(resized_data))
    print(f'Saved transformed data to {output_path}')



def load_data(output_path, split, feature):
    return np.load(f"{data_dir}/{feature}/{feature}_X_{split}.npy")

def save_combo_data(data_dir, split, combos):
    np.save(f"{data_dir}/combo/train_X_combo_{split}.npy", combos)

def main():
    features = ["MTF", "GASF", "GADF", "RP"]
    splits = [1, 2, 3, 4, 5]

    for split in splits:
        combos = []
        for feature in features:
            mtf_train = load_data(data_dir, split, feature)
            combos.append(mtf_train)
        save_combo_data(data_dir, split, np.stack(combos, axis=-1))

