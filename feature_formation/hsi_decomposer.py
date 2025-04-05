import os
import numpy as np
import pandas as pd
from data_load.all_data_load import *
from feature_formation.feat_split_ratio import *
from src.spec_decomp_func import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.cross_decomposition import PLSRegression
from joblib import dump

save_dir = 'data/preprocessed/decomposer'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

labels_train = {'2022': train_pltn_2022} #training the decomposers on just 2022 training data set
X_stack_train, X_train, y_train = create_stacked_dataset({**prep_veg_h_21, **prep_veg_h_22}, rootwt_comb, labels_train)

def PCA_X(X, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on the given dataset.

    This function standardizes the dataset and applies PCA to reduce its dimensionality.
    The number of principal components retained can be specified. If not specified,
    all components are retained.

    Parameters:
    - X (array-like of shape (n_samples, n_features)): The input data to perform PCA on.
      Each row corresponds to a sample, and each column corresponds to a feature.
    - n_components (int, float, None or str): The number of principal components to keep.
      If None, all components are kept. If int, it specifies the exact number of components to retain.
      If float, it represents the fraction of variance that should be retained by the components.

    Returns:
    - pca (object): The PCA object that contains the information about the PCA transformation.
      This includes components, explained variance, and other attributes.
    - scaler (object): The StandardScaler object that was used to standardize the dataset.
    """
    scaler = StandardScaler().fit(X)  # Initialize StandardScaler
    X_scaled = scaler.transform(X)  # Standardize the dataset
    pca = PCA(n_components=n_components)  # Initialize PCA
    pca.fit(X_scaled)  # Fit PCA on the standardized dataset
    return pca, scaler  

pca,scaler = PCA_X(X_train, n_components=3) # We reach above 99% variance explained with 3 components
PCA_decomposer = ImageDecomposer(pca, scaler)
dump(PCA_decomposer, save_dir + '/PCA_decomposer.joblib')

def FA_X(X, n_components=None):
        """
        Perform Factor Analysis (FA) on the given dataset.

        This function standardizes the dataset and applies Factor Analysis to reduce its dimensionality.
        The number of factors retained can be specified. If not specified, all factors are retained.

        Parameters:
        - X (array-like of shape (n_samples, n_features)): The input data to perform FA on.
            Each row corresponds to a sample, and each column corresponds to a feature.
        - n_components (int, None): The number of factors to keep.
            If None, all factors are kept. If int, it specifies the exact number of factors to retain.

        Returns:
        - fa (object): The FactorAnalysis object that contains the information about the FA transformation.
            This includes components, noise variance, and other attributes.
        - scaler (object): The StandardScaler object that was used to standardize the dataset.
        """
        scaler = StandardScaler().fit(X)  # Initialize StandardScaler
        X_scaled = scaler.transform(X)  # Standardize the dataset
        fa = FactorAnalysis(n_components=n_components)  # Initialize Factor Analysis
        fa.fit(X_scaled)  # Fit FA on the standardized dataset
        return fa, scaler

fa,scaler = FA_X(X_train,n_components=3)
FA_decomposer = ImageDecomposer(fa, scaler)
dump(FA_decomposer, save_dir + '/FA_decomposer.joblib')

def ICA_X(X, n_components=None):
    """
    Perform Independent Component Analysis (ICA) on the given dataset.

    This function standardizes the dataset and applies ICA to reduce its dimensionality.
    The number of components retained can be specified. If not specified, all components are retained.

    Parameters:
    - X (array-like of shape (n_samples, n_features)): The input data to perform ICA on.
        Each row corresponds to a sample, and each column corresponds to a feature.
    - n_components (int, None): The number of components to keep.
        If None, all components are kept. If int, it specifies the exact number of components to retain.

    Returns:
    - ica (object): The FastICA object that contains the information about the ICA transformation.
        This includes components_, mixing_, and other attributes.
    - scaler (object): The StandardScaler object that was used to standardize the dataset.
    """
    scaler = StandardScaler().fit(X)  # Initialize StandardScaler
    X_scaled = scaler.transform(X)  # Standardize the dataset
    ica = FastICA(n_components=n_components, random_state=0)  # Initialize ICA
    ica.fit(X_scaled)  # Fit ICA on the standardized dataset
    return ica, scaler

ica,scaler = ICA_X(X_train,n_components=3) 
ICA_decomposer = ImageDecomposer(ica, scaler)
dump(ICA_decomposer, save_dir + '/ICA_decomposer.joblib')

plsr_model = PLSRegression(n_components=3).fit(scaler.transform(X_train), y_train)
PLSR_decomposer = ImageDecomposer(plsr_model, scaler)
dump(PLSR_decomposer, save_dir + '/PLSR_decomposer.joblib')

print("Decomposers saved in " + save_dir)