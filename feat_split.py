#### Splitting the featuresures into training, validation and testing sets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dtype_dict = {'Flight': str}  # To ensure the flight date is read as a string
features = pd.read_csv('data/features_21_22.csv', header=0, dtype=dtype_dict)
############## code for random splitting ###############
test_set_split = 0.25
val_set_split = 0.2
rand_state = 42

train_val_pltn_2021, test_pltn_2021 = train_test_split(np.arange(1,19), test_size=test_set_split, random_state=rand_state)
train_pltn_2021, val_pltn_2021 = train_test_split(train_val_pltn_2021, test_size=val_set_split, random_state=rand_state)

test_ind_2021 = features['Flight'].str.startswith('2021') & features['Plot'].isin(test_pltn_2021)
val_ind_2021 = features['Flight'].str.startswith('2021') & features['Plot'].isin(val_pltn_2021)
train_ind_2021 = features['Flight'].str.startswith('2021') & ~test_ind_2021 & ~val_ind_2021

train_val_pltn_2022, test_pltn_2022 = train_test_split(np.arange(1,89), test_size=test_set_split, random_state=rand_state)
train_pltn_2022, val_pltn_2022 = train_test_split(train_val_pltn_2022, test_size=val_set_split, random_state=rand_state)

test_ind_2022 = features['Flight'].str.startswith('2022') & features['Plot'].isin(test_pltn_2022)
#######################################################

val_ind_2022 = features['Flight'].str.startswith('2022') & features['Plot'].isin(val_pltn_2022)
train_ind_2022 = features['Flight'].str.startswith('2022') & features['Plot'].isin(train_pltn_2022)

test_indices = test_ind_2021 | test_ind_2022
val_indices = val_ind_2021 | val_ind_2022
train_indices = train_ind_2021 | train_ind_2022
train_val_indices = train_indices | val_indices

labels = features[['Flight', 'Plot']]
X = features.iloc[:,2:-1]
y = features['y']

X_train, X_val, X_train_val, X_test = X[train_indices], X[val_indices], X[train_val_indices], X[test_indices]
y_train, y_val, y_train_val, y_test = y[train_indices], y[val_indices], y[train_val_indices],y[test_indices]
labels_train, labels_val, labels_train_val, labels_test = labels[train_indices], labels[val_indices], labels[train_val_indices], labels[test_indices]

print('Train shape: ', X_train.shape)
print('Validation shape: ', X_val.shape)
print('Test shape: ', X_test.shape)
