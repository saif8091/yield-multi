import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error # type: ignore

def gaussian_process_regression_with_tuning(X_train, y_train, X_test, y_test, print_results=True):
    """
    Perform Gaussian Process Regression with hyperparameter tuning on the given training data,
    and predict the output for the test data.

    Parameters:
    - X_train: Training features.
    - y_train: Training targets.
    - X_test: Test features to predict.

    Returns:
    - y_pred: Predicted values for X_test.
    - model: The fitted Gaussian Process Regressor model with the best found hyperparameters.
    """
    # Define different kernel functions
    kernels = [
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3)),
        #C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5),
        #C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), alpha=0.1),
        #C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), periodicity=3.0),
        #C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5))
    ]
    
    # Setup hyperparameter space for tuning
    param_grid = {
        "alpha": [0.05, 0.1, 0.5, 1],
        "kernel": kernels
    }
    # Instantiate a Gaussian Process Regressor
    gpr = GaussianProcessRegressor(n_restarts_optimizer=10)
    
    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(gpr, param_grid=param_grid, cv=5, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model after tuning
    model = grid_search.best_estimator_
    
    # Make predictions using the best found model
    #y_pred = model.predict(X_test)
    if print_results:
        print(f'Model score: {model.score(X_test,y_test)}')
    return model, grid_search

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
        model, grid_search_ = gaussian_process_regression_with_tuning(X_train_sub, y_train, X_test_sub, y_test, print_results=True)
        print(grid_search_.best_params_)
        # Evaluate the model
        r2_train, rmse_train, mape_train, mean_std_train, r2_test, rmse_test, mape_test, mean_std_test, adj_r2_test = evaluate_model_gpr(model, X_train_sub, y_train, X_test_sub, y_test)
        
        # Append results
        results.append({
            "feats": feats,
            #"best_params": grid_search.best_params_,
            "r2_train": r2_train,
            "r2_test": r2_test,

            'mape_train': mape_train * 100,
            'mape_test': mape_test * 100,

            "rmse_train": rmse_train,
            "rmse_test": rmse_test,

            "mean_std_train": mean_std_train,
            "mean_std_test": mean_std_test,
            #'adj_r2_test': adj_r2_test
        })

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df