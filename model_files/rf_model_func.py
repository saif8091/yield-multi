import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

def random_forest_regression_with_tuning(X_train, y_train, X_test, y_test, print_results=True):
    """
    Perform Random Forest Regression with hyperparameter tuning on the given training data,
    and predict the output for the test data.

    Parameters:
    - X_train: Training features.
    - y_train: Training targets.
    - X_test: Test features to predict.
    - y_test: Test targets for evaluation.
    - print_results: Whether to print model score.

    Returns:
    - model: The fitted Random Forest Regressor model with the best found hyperparameters.
    - grid_search: The GridSearchCV object.
    """
    
    # Setup hyperparameter space for tuning
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt', 'log2', None]
    }
    
    # Instantiate a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    
    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model after tuning
    model = grid_search.best_estimator_
    
    if print_results:
        print(f'Model score: {model.score(X_test, y_test)}')
    return model, grid_search

def evaluate_model_rf(model, X_train, y_train, X_val, y_val):
    """
    Evaluate Random Forest model performance.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)

    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    adjusted_r2_val = 1 - ((1 - r2_val) * (n_train - 1) / (n_train - n_features - 1))

    # Calculate feature importance std as a proxy for uncertainty
    feature_importance_std = np.std(model.feature_importances_)

    return r2_train, rmse_train, mape_train, feature_importance_std, r2_val, rmse_val, mape_val, feature_importance_std, adjusted_r2_val

def run_rf_models(X_train, y_train, X_test, y_test, feat_lists):
    """
    Evaluate Random Forest Regression models for each list of features, tracking progress with a progress bar.

    Parameters:
    - X_train: Training features DataFrame.
    - y_train: Training targets Series.
    - X_test: Test features DataFrame.
    - y_test: Test targets Series.
    - feat_lists: List of lists, where each sublist contains feature names to be used in a model.

    Returns:
    - DataFrame with columns for features, best parameters, R^2 and RMSE for training and test sets.
    """
    # Random Forest doesn't require scaling, but we'll keep it for consistency
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    results = []

    # Initialize progress bar
    pbar = tqdm(total=len(feat_lists), desc="Evaluating RF Models")

    for feats in feat_lists:
        # Select features for the current iteration
        X_train_sub = X_train_scaled[feats]
        X_test_sub = X_test_scaled[feats]

        # Train model and get the best estimator
        model, grid_search_ = random_forest_regression_with_tuning(X_train_sub, y_train, X_test_sub, y_test, print_results=True)
        print(grid_search_.best_params_)
        
        # Evaluate the model
        r2_train, rmse_train, mape_train, importance_std_train, r2_test, rmse_test, mape_test, importance_std_test, adj_r2_test = evaluate_model_rf(model, X_train_sub, y_train, X_test_sub, y_test)
        
        # Append results
        results.append({
            "feats": feats,
            "r2_train": r2_train,
            "r2_test": r2_test,
            'mape_train': mape_train * 100,
            'mape_test': mape_test * 100,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "feature_importance_std_train": importance_std_train,
            "feature_importance_std_test": importance_std_test,
        })

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df
