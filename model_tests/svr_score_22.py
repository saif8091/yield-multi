import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_formation.feat_split import *
from model_files.load_feats import *
from model_files.svr_model_func import *

dir_name = 'model_files/model_scores'

# Check if the directory exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)

pca_h = X_train.filter(regex='^pca_h_').columns.tolist()
fa_h = X_train.filter(regex='^fa_h_').columns.tolist()
ica_h = X_train.filter(regex='^ica_h_').columns.tolist()
plsr_h = X_train.filter(regex='^plsr_h_').columns.tolist()
plsr_best_h = X_train.filter(regex='^plsr_best_h_').columns.tolist()

feat_lists_m = [meteo_list + ['vol'],
                meteo_list + ['vol'] + features_dict['cfs_m'],
                meteo_list + ['vol'] + features_dict['mfs_m'],
                meteo_list + ['vol'] + features_dict['micorfs_m']
]

feat_lists_h = [meteo_list + ['vol_lidar'],
                meteo_list + ['vol_lidar'] + features_dict['cfs_h'],
                meteo_list + ['vol_lidar'] + features_dict['mfs_h'],
                meteo_list + ['vol_lidar'] + features_dict['micorfs_h'],
                meteo_list + ['vol_lidar'] + pca_h,
                meteo_list + ['vol_lidar'] + fa_h,
                meteo_list + ['vol_lidar'] + ica_h,
                meteo_list + ['vol_lidar'] + plsr_h
]

feat_list_misc = [meteo_list + features_dict['micorfs_m'],
                  meteo_list + features_dict['micorfs_m']+ ['vol_lidar_m'],  
                  meteo_list + pca_h,
                  meteo_list + pca_h+ ['vol']
] 

results_m = run_svr_models(X_22_train, y_22_train, X_22_test, y_22_test, feat_lists_m)
results_m.to_csv(dir_name+'/svr_scores_22_m.csv', index=False)

results_h = run_svr_models(X_22_train, y_22_train, X_22_test, y_22_test, feat_lists_h)
results_h.to_csv(dir_name+'/svr_scores_22_h.csv', index=False)

results_misc = run_svr_models(X_22_train, y_22_train, X_22_test, y_22_test, feat_list_misc)
results_misc.to_csv(dir_name+'/svr_scores_22_misc.csv', index=False)

print('SVR model scores for 2022 saved in ' + dir_name)
