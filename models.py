import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import warnings
import rasterio
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Add, BatchNormalization, Dense, Conv2D, MaxPooling2D, Input, Activation
from tensorflow.python.ops.gen_batch_ops import Batch
from tqdm import tqdm
import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import time
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, cohen_kappa_score


# InceptionTime model
input_shape = (24, 4, 1)
input_layer = Input(shape=input_shape)

conv1 = Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same')(input_layer)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2,2), padding='same')(conv3)
flatten_cnn = Flatten()(pool3)

tcn1 = Conv2D(64, (7,1), dilation_rate=2, activation='relu', padding='same')(input_layer)
tcn2 = Conv2D(64, (7,1), dilation_rate=4, activation='relu', padding='same')(tcn1)
tcn3 = Conv2D(64, (7,1), dilation_rate=8, activation='relu', padding='same')(tcn2)

conv1x1 = Conv2D(64, (1,1), activation='relu', padding='same')(tcn3)
conv3x3_d2 = Conv2D(64, (3,3), dilation_rate=2, activation='relu', padding='same')(tcn3)
conv5x5_d4 = Conv2D(64, (5,5), dilation_rate=4, activation='relu', padding='same')(tcn3)

concatenated = layers.concatenate([conv1x1, conv3x3_d2, conv5x5_d4])

residual = layers.add([tcn2, tcn3])
flatten_tcn = Flatten()(residual)

concatenated = layers.concatenate([flatten_cnn, flatten_tcn])

output_layer = layers.Dense(8, activation='softmax')(concatenated)

inceptiontime_model = Model(inputs=input_layer, outputs=output_layer)
inceptiontime_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# MultiRocket model
def evaluate(clf, rocket, x_train, y_train, x_valid, y_valid):
    """
    fitness function 
    """
    clf.set_params(**rocket)
    clf.fit(x_train, y_train)
    accuracy = accuracy_score(y_valid, clf.predict(x_valid))
    return accuracy

def multirocket(clf, num_rockets, num_iterations, x_train, y_train, x_valid, y_valid):
    param_grid = {"n_estimators": range(10, 101), "max_depth": range(1, 11)}
    rockets = []
    for i in range(num_rockets):
        rocket = {"n_estimators": random.randint(10, 100), "max_depth": random.randint(1, 10)}
        rockets.append(rocket)
    for i in range(num_iterations):
        for j in range(num_rockets):
            random_search = RandomizedSearchCV(clf, param_grid, n_iter=1, cv=5, scoring=evaluate, refit=True)
            random_search.fit(x_train, y_train)
            rockets[j] = random_search.best_params_
    return max(rockets, key=lambda x: evaluate(clf, x, x_train, y_train, x_valid, y_valid))

clf = RandomForestClassifier()
num_rockets = 3
num_iterations = 100
best_params = multirocket(clf, num_rockets, num_iterations, x_train, y_train, x_valid, y_valid)

# Random Forest model
tuned_parameters = {'n_estimators': [10], 'max_depth': [20]}
X = np.concatenate((train_X, valid_X), axis=0)
y = np.concatenate((train_y, valid_y), axis=0)
mytestfold = [-1] * train_y.size + [0] * valid_y.size
ps = PredefinedSplit(test_fold=mytestfold)
clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=tuned_parameters, cv=ps, n_jobs=-1, verbose=2)
clf.fit(X, y)

# ResNet model
def residual_block(X_start, filters, name, reduce=False, res_conv2d=False):
    nb_filters_1, nb_filters_2, nb_filters_3 = filters
    strides_1 = [2, 2] if reduce else [1, 1]
    X = Conv2D(filters=nb_filters_1, kernel_size=[1, 1], strides=strides_1, padding='same', name=name)(X_start)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=nb_filters_2, kernel_size=[3, 3], strides=[1, 1], padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=nb_filters_3, kernel_size=[1, 1], strides=[1, 1], padding='same')(X)
    X = BatchNormalization()(X)
    if res_conv2d:
        X_res = Conv2D(filters=nb_filters_3, kernel_size=[1, 1], strides=strides_1, padding='same')(X_start)
        X_res = BatchNormalization()(X_res)
    else:
        X_res = X_start
    X = Add()([X, X_res])
    X = Activation('relu')(X)
    return X

def resnet50(input_shape, nb_classes):
    X_input = Input(shape=input_shape)
    X = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding='same', name='conv1')(X_input)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D([3, 3], strides=[2, 2])(X)
    X = residual_block(X, filters=[64, 64, 256], name='conv2_a', reduce=False, res_conv2d=True)
    X = residual_block(X, filters=[64, 64
