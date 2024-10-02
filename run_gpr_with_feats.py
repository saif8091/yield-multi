import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error # type: ignore
from feat_split import *

meteo_list = ['gdd','gdd_harvest','evap','evap_harvest']
ref_m_dji = ['ref_m_mean_1', 'ref_m_mean_2', 'ref_m_mean_3', 'ref_m_mean_4'] 
ref_m = X_train.filter(regex='^ref_m_').columns.tolist()
cfs_m = ['vi_osavi_m_mean', 'vi_gi_m_mean']
mfs_m = ['vi_pbi_m_mean', 'vi_msr_m_mean', 'vi_tcari_m_mean']
micorfs_m = ['vi_pbi_m_mean', 'vi_psnd_m_mean', 'vi_msr_m_mean', 'vi_tcari_m_mean', 'vi_gi_m_mean']

ref_h1_mica = ['ref_h_mean_11', 'ref_h_mean_24', 'ref_h_mean_40', 'ref_h_mean_47', 'ref_h_mean_66']
ref_h2 = ['ref_h_mean_24', 'ref_h_mean_32', 'ref_h_mean_40', 'ref_h_mean_47', 'ref_h_mean_66']
cfs_h = ['vi_psnd_h_mean', 'vi_gi_h_mean']
mfs_h = ['vi_pbi_h_mean', 'vi_tcari_h_mean', 'vi_mcari_h_mean', 'vi_ndre_h_mean']
micorfs_h = ['vi_pbi_h_mean', 'vi_psnd_h_mean', 'vi_tcari_h_mean', 'vi_mcari_h_mean', 'vi_gi_h_mean','vi_ndre_h_mean']
pca_h = X_train.filter(regex='^pca_h_').columns.tolist()
fa_h = X_train.filter(regex='^fa_h_').columns.tolist()
ica_h = X_train.filter(regex='^ica_h_').columns.tolist()
plsr_h = X_train.filter(regex='^plsr_h_').columns.tolist()
plsr_best_h = X_train.filter(regex='^plsr_best_h_').columns.tolist()

feat_lists = [meteo_list + ['vol_lidar'],
            meteo_list + ['vol'],
            meteo_list + ['vol'] + ref_m_dji,
            meteo_list + ['vol'] + ref_m,
            meteo_list + ['vol'] + cfs_m,
            meteo_list + ['vol'] + mfs_m,
            meteo_list + ['vol'] + micorfs_m,

            meteo_list + ['vol_lidar'] + ref_h1_mica,
            meteo_list + ['vol_lidar'] + ref_h2,
            meteo_list + ['vol_lidar'] + cfs_h,
            meteo_list + ['vol_lidar'] + mfs_h,
            meteo_list + ['vol_lidar'] + micorfs_h,
            meteo_list + ['vol_lidar'] + pca_h,
            meteo_list + ['vol_lidar'] + fa_h,
            meteo_list + ['vol_lidar'] + ica_h,
            meteo_list + ['vol_lidar'] + plsr_h,

            meteo_list + ['vol_lidar'] + micorfs_m,
            meteo_list + ['vol'] + micorfs_h    
]

def gaussian_process_regression(X_train, y_train):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
    return GaussianProcessRegressor(kernel=kernel, alpha=0.5, n_restarts_optimizer=10).fit(X_train, y_train)

def evaluate_model_gpr(model, X_train, y_train, X_val, y_val):
    y_train_pred, y_train_std = model.predict(X_train, return_std=True)
    y_val_pred, y_val_std = model.predict(X_val, return_std=True)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)

    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    adjusted_r2_val = 1 - ((1 - r2_val) * (n_train - 1) / (n_train - n_features - 1))

    return r2_train, rmse_train, mape_train, y_train_std.mean(), r2_val, rmse_val, mape_val, y_val_std.mean(), adjusted_r2_val
#model, grid = gaussian_process_regression_with_tuning(X_train_val[feat_names], y_train_val, X_test[feat_names], y_test)

def run_gpr_models(X_train, y_train, X_test, y_test, feat_lists):
    """
    Evaluate Gaussian Process Regression models for each list of features, tracking progress with a progress bar.

    Parameters:
    - X_train: Training features DataFrame.
    - y_train: Training targets Series.
    - X_test: Test features DataFrame.
    - y_test: Test targets Series.
    - feat_lists: List of lists, where each sublist contains feature names to be used in a model.

    Returns:
    - DataFrame with columns for features, best parameters, R^2 and RMSE for training and test sets, and mean standard deviation of predictions.
    """
    scaler=StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    results = []

    # Initialize progress bar
    pbar = tqdm(total=len(feat_lists), desc="Evaluating GPR Models")

    for feats in feat_lists:
        # Select features for the current iteration
        X_train_sub = X_train_scaled[feats]
        X_test_sub = X_test_scaled[feats]

        # Train model and get the best estimator
        model = gaussian_process_regression(X_train_sub, y_train) #gaussian_process_regression_with_tuning(X_train_sub, y_train, X_test_sub, y_test, print_results=False)
        
        # Evaluate the model
        r2_train, rmse_train, mape_train, mean_std_train, r2_test, rmse_test, mape_test, mean_std_test, adj_r2_test = evaluate_model_gpr(model, X_train_sub, y_train, X_test_sub, y_test)
        
        # Append results
        results.append({
            "feats": feats,
            #"best_params": grid_search.best_params_,
            "r2_train": r2_train,
            "rmse_train": rmse_train,
            'mape_train': mape_train,
            "mean_std_train": mean_std_train,
            "r2_test": r2_test,
            "rmse_test": rmse_test,
            'mape_test': mape_test,
            "mean_std_test": mean_std_test,
            'adj_r2_test': adj_r2_test
        })

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

results = run_gpr_models(X_train_val, y_train_val, X_test, y_test, feat_lists)
results.to_csv('gpr_scores.csv', index=False)