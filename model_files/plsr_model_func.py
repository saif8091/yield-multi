import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

def plsr_regression_with_tuning(X_train, y_train, X_test, y_test, print_results=True):
    """
    Perform Partial Least Squares Regression with hyperparameter tuning on the given training data,
    and predict the output for the test data.

    Parameters:
    - X_train: Training features.
    - y_train: Training targets.
    - X_test: Test features to predict.
    - y_test: Test targets for evaluation.
    - print_results: Whether to print model score.

    Returns:
    - model: The fitted PLSR model with the best found hyperparameters.
    - grid_search: The GridSearchCV object.
    """
    
    # Setup hyperparameter space for tuning
    # Number of components should not exceed min(n_samples, n_features) - 1
    max_components = min(X_train.shape[0], X_train.shape[1]) - 1
    n_components_range = list(range(1, min(max_components + 1, 21)))  # Cap at 20 components
    
    param_grid = {
        "n_components": n_components_range,
        "scale": [True, False],
        "max_iter": [500, 1000]
    }
    
    # Instantiate a PLS Regressor
    plsr = PLSRegression()
    
    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(plsr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model after tuning
    model = grid_search.best_estimator_
    
    if print_results:
        print(f'Model score: {model.score(X_test, y_test)}')
    return model, grid_search

def evaluate_model_plsr(model, X_train, y_train, X_val, y_val):
    """
    Evaluate PLSR model performance.
    """
    y_train_pred = model.predict(X_train).flatten()
    y_val_pred = model.predict(X_val).flatten()

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)

    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    adjusted_r2_val = 1 - ((1 - r2_val) * (n_train - 1) / (n_train - n_features - 1))

    # Use explained variance as uncertainty proxy
    explained_variance = np.var(y_val_pred) / np.var(y_val)

    return r2_train, rmse_train, mape_train, explained_variance, r2_val, rmse_val, mape_val, explained_variance, adjusted_r2_val

def run_plsr_models(X_train, y_train, X_test, y_test, feat_lists):
    """
    Evaluate PLSR models for each list of features, tracking progress with a progress bar.

    Parameters:
    - X_train: Training features DataFrame.
    - y_train: Training targets Series.
    - X_test: Test features DataFrame.
    - y_test: Test targets Series.
    - feat_lists: List of lists, where each sublist contains feature names to be used in a model.

    Returns:
    - DataFrame with columns for features, best parameters, R^2 and RMSE for training and test sets.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    results = []

    # Initialize progress bar
    pbar = tqdm(total=len(feat_lists), desc="Evaluating PLSR Models")

    for feats in feat_lists:
        # Select features for the current iteration
        X_train_sub = X_train_scaled[feats]
        X_test_sub = X_test_scaled[feats]

        # Train model and get the best estimator
        model, grid_search_ = plsr_regression_with_tuning(X_train_sub, y_train, X_test_sub, y_test, print_results=True)
        print(grid_search_.best_params_)
        
        # Evaluate the model
        r2_train, rmse_train, mape_train, var_explained_train, r2_test, rmse_test, mape_test, var_explained_test, adj_r2_test = evaluate_model_plsr(model, X_train_sub, y_train, X_test_sub, y_test)
        
        # Append results
        results.append({
            "feats": feats,
            "r2_train": r2_train,
            "r2_test": r2_test,
            'mape_train': mape_train * 100,
            'mape_test': mape_test * 100,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "var_explained_train": var_explained_train,
            "var_explained_test": var_explained_test,
        })

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df
