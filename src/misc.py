''' Contains a number of miscellaneous functions
'''
import numpy as np
import pandas as pd
from datetime import datetime

def DAP(date_array, harvest_date):
    """
    Calculate the difference in days between each element of the date_array and the harvest_date.

    Parameters:
    - date_array (list of str): Array of date strings in the format 'YYYYMMDD'.
    - harvest_date (str): harvest date string in the format 'YYYYMMDD'.

    Returns:
    - np.ndarray: Array of numbers representing the difference in days.
    """
    # Convert harvest_date to datetime object
    ref_date = datetime.strptime(harvest_date, '%Y%m%d')
    
    # Calculate the difference in days for each date in date_array
    day_differences = [abs((datetime.strptime(date, '%Y%m%d') - ref_date).days) for date in date_array]
    
    return np.array(day_differences)

def mask_out(img, mask):
    '''multiplying each channel of the image with the mask'''
    return img * mask[..., np.newaxis]

def m_spec_with_zero(im):
    '''function to find mean including the zeros'''
    _,_,nb=im.shape
    return np.reshape(im,(-1,nb)).mean(0)

def m_spec(im):
    '''function to find mean ignoring the zeros'''
    _,_,nb=im.shape
    im_c = im.copy()
    im_c[im_c==0]=np.nan
    return np.nanmean(np.reshape(im_c,(-1,nb)),0)

def m_spec_with_mask(im,prep_im):
    '''This function calculates the mean across each channels for im ignoring the values with zero prep_im'''
    _,_,nb=im.shape
    im_c = im.copy()
    im_c[prep_im==0]=np.nan
    return np.nanmean(np.reshape(im_c,(-1,nb)),0)

def increment_column_suffix(df):
    """
    This function increments the numeric suffix in each column name of a DataFrame by 1.
    For example, if a column name is 'feature_1', it will be renamed to 'feature_2'.
    If a column name does not end with a number, it remains unchanged.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column names are to be updated.

    Returns:
    df (pandas.DataFrame): The DataFrame with updated column names.
    """
    new_columns = []
    for col in df.columns:
        parts = col.split('_')
        if parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            new_columns.append('_'.join(parts))
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df

class CustomStandardScaler:
    def __init__(self):
        """
        Initialize the CustomStandardScaler.
        """
        self.means = None
        self.stds = None
        self.columns = None

    def fit(self, X):
        """
        Compute the mean and standard deviation for the columns to be scaled.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to fit the scaler on.

        Returns:
        - self: The fitted scaler.
        """
        self.means = X.mean()
        self.stds = X.std()
        self.columns = X.columns
        return self

    def transform(self, X):
        """
        Scale the columns of X using the computed mean and standard deviation.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to transform.

        Returns:
        - X_scaled (pd.DataFrame): The DataFrame with scaled columns.
        """
        # Find the common columns between the fitted DataFrame and the input DataFrame
        common_columns = [col for col in self.columns if col in X.columns] # type: ignore
        
        # Check if there are any common columns
        if not common_columns:
            raise ValueError("No matching columns found between the input DataFrame and the DataFrame used for fitting.")
        
        X_scaled = X.copy()
        X_scaled[common_columns] = (X[common_columns] - self.means[common_columns]) / self.stds[common_columns] # type: ignore
        return X_scaled

    def fit_transform(self, X):
        """
        Fit the scaler on X and then transform X.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to fit and transform.

        Returns:
        - X_scaled (pd.DataFrame): The DataFrame with scaled columns.
        """
        return self.fit(X).transform(X)