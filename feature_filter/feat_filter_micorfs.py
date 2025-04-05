import numpy as np
import pandas as pd
from feature_filter.feature_selection import FilterBasedFeatureSelection
from feature_formation.feat_split import *

dir_name = 'feature_filter/filtered_features'
print('Performing correlation and mutual information based feature selection')
micorfs_m = FilterBasedFeatureSelection(X_22_train.filter(regex='_m_'),y_22_train,handle='mi_cor')
selected_features_micorfs_m = micorfs_m.select_based_on_threshold(min_r2=0.75, r2_thresh=0.80)

open(dir_name + '/micorfs_m.txt', 'w').writelines(f"{item}\n" for item in selected_features_micorfs_m)
print(f'Number of features selected for micorfs_m: {len(selected_features_micorfs_m)}')

micorfs_h = FilterBasedFeatureSelection(X_22_train.filter(regex='(^vi.*_h_)|(ref_h_)'),y_22_train,handle='mi')
selected_features_micorfs_h = micorfs_h.select_based_on_threshold(min_r2=0.75, r2_thresh=0.80)

open(dir_name + '/micorfs_h.txt', 'w').writelines(f"{item}\n" for item in selected_features_micorfs_h)
print(f'Number of features selected for micorfs_m: {len(selected_features_micorfs_h)}')