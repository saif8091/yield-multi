from feature_formation.feat_split import *
from src.misc import CustomStandardScaler
from model_files.gpr_model_func import *
from model_files.load_feats import *
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations
from tqdm import tqdm
from random import sample

feat_m = meteo_list + ['vol'] + features_dict['micorfs_m']
pca_h = X_train.filter(regex='^pca_h_').columns.tolist()
feat_h = meteo_list + ['vol_lidar'] + pca_h 

feat_m_l = meteo_list + ['vol_lidar_m'] + features_dict['micorfs_m']
feat_h_m = meteo_list + ['vol'] + pca_h

def train_model(X_train, y_train, X_test, y_test, feat):
    scaler = CustomStandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model, grid = gaussian_process_regression_with_tuning(X_train_scaled[feat], y_train, X_test_scaled[feat], y_test, print_results=True)
    return model, scaler, X_train_scaled, X_test_scaled, grid

def generate_train_test_splits_max_split(arr, n, max_splits=10):
    """
    Generate all possible train-test splits for the given array and number of training points,
    with a maximum limit on the number of splits.

    Parameters:
    - arr (np.ndarray): The array to split.
    - n (int): The number of data points in the training set.
    - max_splits (int): The maximum number of splits to generate for each train size.

    Returns:
    - list: A list of tuples, where each tuple contains a train set and a test set.
    """
    splits = []
    for i in range(n, -1, -1):
        all_combinations = list(combinations(arr, i))
        if len(all_combinations) > max_splits:
            selected_combinations = sample(all_combinations, max_splits)
        else:
            selected_combinations = all_combinations
        for train_indices in selected_combinations:
            train_set = np.array(train_indices)
            test_set = np.array([x for x in arr if x not in train_set])
            splits.append((train_set, test_set))
    return splits

def generate_train_test_splits(arr, n):
    """
    Generate all possible train-test splits for the given array and number of training points.

    Parameters:
    - arr (np.ndarray): The array to split.
    - n (int): The number of data points in the training set.

    Returns:
    - list: A list of tuples, where each tuple contains a train set and a test set.
    """
    splits = []
    for i in range(n, -1, -1):
        for train_indices in combinations(arr, i):
            train_set = np.array(train_indices)
            test_set = np.array([x for x in arr if x not in train_set])
            splits.append((train_set, test_set))
    return splits

def generate_random_train_test_splits(arr, n, random_state=None):
    """
    Generate random train-test splits for the given array and number of training points,
    including all values below n up to 0.

    Parameters:
    - arr (np.ndarray): The array to split.
    - n (int): The maximum number of data points in the training set.
    - random_state (int, optional): The random state for reproducibility.

    Returns:
    - list: A list of tuples, where each tuple contains a train set and a test set.
    """
    splits = []
    for i in range(n, -1, -1):
        # Perform the train-test split
        if i > 0:
            train_set, test_set = train_test_split(arr, train_size=i, random_state=random_state)
        else:
            train_set = []
            test_set = arr
        splits.append((train_set, test_set))
    return splits

def generate_train_test_indices(features, splits):
    """
    Generate test_ind_2021 and train_ind_2021 indices for each split.

    Parameters:
    - features (pd.DataFrame): DataFrame containing the features with 'Flight' and 'Plot' columns.
    - splits (list): List of tuples, where each tuple contains a train set and a test set.

    Returns:
    - list: A list of tuples, where each tuple contains test_ind_2021 and train_ind_2021 indices.
    """
    indices = []
    for train_set, test_set in splits:
        test_ind_2021 = features['Flight'].str.startswith('2021') & features['Plot'].isin(test_set)
        train_ind_2021 = features['Flight'].str.startswith('2021') & features['Plot'].isin(train_set)
        indices.append((test_ind_2021, train_ind_2021))
    return indices

def evaluate_model_gpr(model, X_train, y_train, X_val, y_val):
    y_train_pred, y_train_std = model.predict(X_train, return_std=True)
    y_val_pred, y_val_std = model.predict(X_val, return_std=True)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)

    return r2_train, rmse_train, mape_train, y_train_std.mean(), r2_val, rmse_val, mape_val, y_val_std.mean()

def train_and_evaluate(feat, n, X_22_train=X_22_train, y_22_train=y_22_train, X_22_test=X_22_test, y_22_test=y_22_test, X=X, y=y, all_features = features, random_state=42, split_all=False):
    results = []
    if split_all:
        splits = generate_train_test_splits_max_split(np.arange(1,19), n)
    else:
        splits = generate_random_train_test_splits(np.arange(1,19), n, random_state=random_state)
    indices = generate_train_test_indices(all_features, splits)

    for test_ind_2021, train_ind_2021 in tqdm(indices, desc="Processing splits"):
        X_21_train = X[train_ind_2021]
        y_21_train = y[train_ind_2021]
        X_21_test = X[test_ind_2021]
        y_21_test = y[test_ind_2021]

        X_train_combined = pd.concat([X_21_train, X_22_train])
        y_train_combined = pd.concat([y_21_train, y_22_train])

        X_test_combined = pd.concat([X_21_test, X_22_test])
        y_test_combined = pd.concat([y_21_test, y_22_test])

        model, scaler, X_train_scaled, X_test_scaled, grid = train_model(X_train_combined, y_train_combined, X_test_combined, y_test_combined, feat)

        r2_train, rmse_train, mape_train, y_train_std_mean, r2_val_22, rmse_val_22, mape_val_22, y_val_std_mean_22 = evaluate_model_gpr(model, X_train_scaled[feat], y_train_combined,scaler.transform(X_22_test)[feat], y_22_test)
        r2_test_all, rmse_test_all, mape_test_all, y_test_std_all, r2_val_21, rmse_val_21, mape_val_21, y_val_std_mean_21 = evaluate_model_gpr(model, X_test_scaled[feat], y_test_combined, scaler.transform(X_21_test)[feat], y_21_test)

        results.append({
            'n_train_2021_plots': train_ind_2021.sum()/4,
            'r2_train': r2_train,
            'r2_test_all': r2_test_all,
            'r2_test_22': r2_val_22,
            'r2_test_21': r2_val_21,
    
            'mape_train': mape_train,
            'mape_test_all': mape_test_all, 
            'mape_test_22': mape_val_22,
            'mape_test_21': mape_val_21,
        
            'rmse_train': rmse_train,
            'rmse_test_all': rmse_test_all,
            'rmse_test_22': rmse_val_22,
            'rmse_test_21': rmse_val_21,

            'y_train_std_mean': y_train_std_mean,
            'y_test_std_mean_all': y_test_std_all,  
            'y_test_std_mean_22': y_val_std_mean_22,
            'y_test_std_mean_21': y_val_std_mean_21
        })

    return pd.DataFrame(results)

print('Starting with m')    
scores_all_m = train_and_evaluate(feat_m, n=17, random_state=42,split_all=True)
scores_all_m.to_csv('model_test_21/model_scores/gpr_scores_21_m.csv')
print('Done with m')
print('Starting with h')
scores_all_h = train_and_evaluate(feat_h, n=17, random_state=42,split_all=True) 
scores_all_h.to_csv('model_test_21/model_scores/gpr_scores_21_h.csv')   
print('Done with h')
print('Starting with m_l')
scores_all_m_l = train_and_evaluate(feat_m_l, n=17, random_state=42,split_all=True)
scores_all_m_l.to_csv('model_test_21/model_scores/gpr_scores_21_m_l.csv')
print('Done with m_l')
print('Starting with h_m')
scores_all_h_m = train_and_evaluate(feat_h_m, n=17, random_state=42,split_all=True)
scores_all_h_m.to_csv('model_test_21/model_scores/gpr_scores_21_h_m.csv')