import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

# Functions for stacking images
def remove_rows_all_nan(array):
    """
    Remove rows from a numpy array where all values in the row are NaN.

    Parameters:
    - array (numpy.ndarray): The input array from which rows with all NaN values should be removed.

    Returns:
    - numpy.ndarray: A new array with rows containing all NaN values removed.
    """
    # Identify rows where all values are NaN
    mask = ~np.isnan(array).all(axis=1)
    
    # Return the array without rows that have all NaN values
    return array[mask]

def single_rioimage_to_X(image):
    """
    Convert a single rioxarray image to a 1D array for input to PCA.

    Parameters:
    - image (rioxarry): shape (ch,y,x).

    Returns:
    - numpy.ndarray: 2D array of shape (n_samples, n_features).
    """
    return remove_rows_all_nan(image.values.reshape(image.shape[0],-1).T)

def create_stacked_dataset(image_dict,y_comb, labels=None):
    """
    Create a stacked dataset from a nested dictionary of rioxarray images, optionally filtered by labels.

    Parameters:
    - image_dict (dict): Nested dictionary with structure {Flight: {Plot: rioxarray image}}.
    - labels (pd.DataFrame, optional): DataFrame with columns 'Flight' and 'Plot' to filter images.

    Returns:
    - X (numpy.ndarray): 2D array of shape (N, ch) after removing rows with all NaN values.
    - y (numpy.ndarray): 1D array of shape (N,) with the corresponding labels.
    """
    X_list = []  # To hold the reshaped arrays from images
    X_mean = []  # To hold the mean of each image
    y=[]

    # Iterate through the nested dictionary
    for flight, plots in image_dict.items():
        year = flight[:4]  # Extract the year from the flight key
        for plot, image in plots.items():
            # If labels are provided, skip images not in labels
            if labels is not None:
                # Check if the year is in labels and if the plot is in the array for that year
                if year not in labels or plot not in labels[year]:
                    continue
            
            # Convert image to 2D array and append to list
            reshaped_image = single_rioimage_to_X(image)
            X_list.append(reshaped_image)
            X_mean.append(np.nanmean(image, axis=(1,2)))
            y.append(y_comb[flight][plot-1])

    # Stack all 2D arrays to form X
    X_stacked = np.vstack(X_list)
    X_mean = np.vstack(X_mean)

    return X_stacked,X_mean, np.array(y)

class ImageDecomposer:
    def __init__(self, model_decomp, model_scale, whitening_matrix=None, input_img=False):
        """
        Initialize the ImageDecomposer object with a decomposition and scaling model.

        Parameters:
        - model_decomp (object): The decomposition model object (e.g., PCA, ICA).
        - model_scale (object): The scaling model object (e.g., StandardScaler).
        - whitening_matrix (numpy.ndarray, optional): The whitening matrix to apply before decomposition (for MNF transformations only).
        - input_img (bool, optional): Whether the input to the transform method is a rioxarray image (True) or a 2D array (False).
        """
        self.model_decomp = model_decomp
        self.model_scale = model_scale
        self.whitening_matrix = whitening_matrix
        self.input_img = input_img

    def transform(self, img):
        """
        Decompose a single rioxarray image using the initialized decomposition and scaling models.

        Parameters:
        - img (rioxarray): The input image to decompose.

        Returns:
        - numpy.ndarray: The decomposed image.
        """
        if self.input_img:
            # Convert the image to a 2D array
            X = single_rioimage_to_X(img)
        else:
            # Assuming img is the input image or array
            if img.ndim == 1:
                # Single sample case
                X = img.reshape(1, -1)
            else:
                # Multiple samples case
                X = img.reshape(img.shape[0], -1)

# Now X is robust for both single sample and multiple samples
        
        # Standardize the data
        X_scaled = self.model_scale.transform(X)
        
        if self.whitening_matrix is not None:
            X_scaled = np.dot(X_scaled, self.whitening_matrix)

        # Perform decomposition
        X_decomposed = self.model_decomp.transform(X_scaled)
        
        return X_decomposed