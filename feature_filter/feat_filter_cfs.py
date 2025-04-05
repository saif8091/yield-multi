import os
import numpy as np
import pandas as pd
from feature_filter.feature_selection import FilterBasedFeatureSelection
from feature_formation.feat_split import *

dir_name = 'feature_filter/filtered_features'

# Check if the directory exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)

print('Performing correlation based feature selection')

cfs_m = FilterBasedFeatureSelection(X_22_train.filter(regex='_m_'),y_22_train,handle='cor')
selected_features_cfs_m = cfs_m.select_based_on_threshold(min_r2=0.10, r2_thresh=0.80)

open(dir_name+'/cfs_m.txt', 'w').writelines(f"{item}\n" for item in selected_features_cfs_m)
print(f'Number of features selected for cfs_m: {len(selected_features_cfs_m)}')

cfs_h = FilterBasedFeatureSelection(X_22_train.filter(regex='(^vi.*_h_)|(ref_h_)'),y_22_train,handle='cor')
selected_features_cfs_h = cfs_h.select_based_on_threshold(min_r2=0.10, r2_thresh=0.80)

open(dir_name+'/cfs_h.txt', 'w').writelines(f"{item}\n" for item in selected_features_cfs_h)
print(f'Number of features selected for cfs_m: {len(selected_features_cfs_h)}')