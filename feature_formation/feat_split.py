#### Splitting the featuresures into training, validation and testing sets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_formation.feat_split_ratio import *

dtype_dict = {'Flight': str}  # To ensure the flight date is read as a string
features = pd.read_csv('data/preprocessed/features_21_22.csv', header=0, dtype=dtype_dict)

test_ind_2021 = features['Flight'].str.startswith('2021') & features['Plot'].isin(test_pltn_2021)
train_ind_2021 = features['Flight'].str.startswith('2021') & ~test_ind_2021

train_ind_2022 = features['Flight'].str.startswith('2022') & features['Plot'].isin(train_pltn_2022)
test_ind_2022 = features['Flight'].str.startswith('2022') & features['Plot'].isin(test_pltn_2022)

test_indices = test_ind_2021 | test_ind_2022
train_indices = train_ind_2021 | train_ind_2022

labels = features[['Flight', 'Plot']]
X = features.iloc[:,2:-1]
y = features['y']

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
labels_train, labels_test = labels[train_indices], labels[test_indices]

X_21 = X[train_ind_2021 | test_ind_2021]
y_21 = y[train_ind_2021 | test_ind_2021]
labels_21 = labels[train_ind_2021 | test_ind_2021]
X_21_train, X_21_test = X[train_ind_2021], X[test_ind_2021]
y_21_train, y_21_test = y[train_ind_2021], y[test_ind_2021]
labels_21_train, labels_21_test = labels[train_ind_2021], labels[test_ind_2021]

X_22 = X[train_ind_2022 | test_ind_2022]
y_22 = y[train_ind_2022 | test_ind_2022]
X_22_train, X_22_test = X[train_ind_2022], X[test_ind_2022]
y_22_train, y_22_test = y[train_ind_2022], y[test_ind_2022]
labels_22_train, labels_22_test = labels[train_ind_2022], labels[test_ind_2022]

print('Train shape: ', X_train.shape)
print('Test shape: ', X_test.shape)