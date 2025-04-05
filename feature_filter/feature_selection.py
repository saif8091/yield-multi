import numpy as np
import pandas as pd
import itertools
from sklearn.feature_selection import r_regression
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

def MI(X,y,reg=True):
    '''
    This function calculates muutal information non-parametrically (Ross, 2014)
    If reg = True than assumes regressor otherwise classifier
    '''
    if reg:
        IG = mutual_info_regression(X,y,random_state=0)
    else:
        IG = mutual_info_classif(X,y,random_state=0) 
    return  pd.Series(IG,index=X.columns)

def calculate_self_MI(X):
    '''This function generates a data frame of self mutual information values for each feature'''
    su_values = {}
    for col in X.columns:
        su = MI(X,X[col])
        su_values[col] = su
    return pd.DataFrame(su_values)

class FilterBasedFeatureSelection:
    '''
    Example usage:
    cfs = FilterBasedFeatureSelection(X_train[tex_feat],y_train)
    cfs.exhaustive_search()
    '''
    def __init__(self, X, y, handle='cor', reg=True):
        """
        Initialize the FilterBasedFeatureSelection object with feature matrix X and target vector y.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        handle (str): Method to calculate rff_mat and rcf_series. 'cor' for correlation, 
        'mi' for mutual information or mi_cor for mutual information to identify varibles
        related to y and intercorrelation to identify related variables. Default is 'cor'.
        reg (bool): True if regression problem, False if classification. Default is True.
        """
        self.X = X
        self.y = y
        self.handle = handle
        self.reg = reg
        self.all_features = X.columns
        self.selected_features = []   #initialise with max r2 value
        self.merit_score = float('-inf')

        if handle == 'cor':
            self.rff_mat = X.corr()**2    # Generates R2 matrix for X
            self.rcf_series = pd.Series(r_regression(X,y),index=X.columns)**2   # generates R2 series for each features with y
        elif handle == 'mi':
            self.rff_mat = calculate_self_MI(X)
            self.rcf_series = MI(X,y,reg=reg)
        elif handle == 'mi_cor':
            self.rff_mat = X.corr()**2    # Generates R2 matrix for X
            self.rcf_series = MI(X,y,reg=reg)
        else:
            raise ValueError("Invalid handle. Expected 'cor', 'mi' or 'mi_cor'.")

    def merit_calculation(self, feature_subset):
        """
        Measure the merit of the selected subset of features.
        Formula from Hall et.al(2000)
        Parameters:
        feature_subset (set): A set of feature names or indices.

        Returns:
        float: Merit score.
        """
        k = len(feature_subset)
        rcf_mean = self.rcf_series[feature_subset].mean()
        rff_mat = self.rff_mat[feature_subset].loc[feature_subset]
        rff_mean = rff_mat.where(np.triu(np.ones(rff_mat.shape), k=0).astype(bool)).stack().values.mean() # extracting the elements of upper triangle
        
        return k*rcf_mean/np.sqrt(k+k*(k-1)*rff_mean)

    def exhaustive_search(self):
        """
        Execute the exhaustive search to find the feature subset with the maximum merit score.

        Returns:
        set: Set of feature names that maximize the merit score.
        """
        all_features = self.all_features
        best_subset = self.selected_features
        best_merit_score = self.merit_score

        pbar = tqdm(total=len(all_features), desc='Checking combinations')
        # Generate all possible feature subsets using itertools
        for r in range(1, len(all_features) + 1):
            
            for subset in itertools.combinations(all_features, r):
                merit_score = self.merit_calculation(set(subset))
                #print(subset,merit_score)
                if merit_score > best_merit_score:
                    best_subset = set(subset)
                    best_merit_score = merit_score
            pbar.update(1)
        pbar.close()
        self.selected_features = best_subset
        self.merit_score = best_merit_score

        return best_subset
    
    def best_first_search_selection(self, max_failed_attempts=3):
        """
        Perform feature selection using the Best-First Search algorithm to find the feature subset with the highest merit score.

        Returns:
        set: The selected feature subset.
        """
        best_subset = set()
        best_merit = float('-inf')

        # Initialize the priority queue with individual features
        priority_queue = [(self.rcf_series[feature], {feature}) for feature in self.X.columns]
        # Sort the priority queue by RCF score
        priority_queue.sort(key=lambda x: x[0])

        consecutive_failed_attempts = 0

        while priority_queue:
            current_rcf_score, current_subset = priority_queue.pop()
            # Calculate merit score for the current subset
            current_merit = self.merit_calculation(current_subset)
            print(current_subset, current_merit)
            if current_merit > best_merit:
                best_merit = current_merit
                best_subset = current_subset
                consecutive_failed_attempts = 0
            else:
                consecutive_failed_attempts += 1

            if consecutive_failed_attempts >= max_failed_attempts:
                break    # stop the search

            # Generate potential extensions
            potential_extensions = self.all_features - current_subset

            # Consider adding one more feature to the current subset
            for feature in potential_extensions:
                new_subset = current_subset | {feature}
                new_rcf_score = self.rcf_series[list(new_subset)].mean()

                # Priority is based on RCF score
                priority = new_rcf_score
                priority_queue.append((priority, new_subset))

            # Sort the priority queue by RCF score
            priority_queue.sort(key=lambda x: x[0])

        self.selected_features = best_subset
        self.merit_score = best_merit
        return best_subset
    
    def select_based_on_threshold(self, min_r2 = 0.2, r2_thresh=0.8):
        """
        Selects feature set with below threshold R-squared values with each other.

        Parameters:
        - min_r2 (float): If handle 'cor' : 
            Minimum R-squared value for a feature with y. Default is 0.2.
            If handle 'SU' :
            Percentile for minimum symmetrical uncertainty value for a feature with y. Default is 0.2.
        - r2_thresh (float): If handle 'cor': 
            Threshold for R-squared values. Default is 0.8.
            If handle 'SU' :
            Threshold for symmetrical uncertainty values with each feature. Default is 0.8.

        Returns:
        - Set of selected features.
        """
        # Initialize cor_matrix by copying rff_mat
        cor_matrix = self.rff_mat.copy()
        np.fill_diagonal(cor_matrix.values, 0) # making the diagonal zero
        
        #'''
        if self.handle == 'mi':
            # convert min_r2 and r2_thresh to percentiles for symmetrical uncertainty
            min_r2 = np.percentile(self.rcf_series, min_r2*100)
            r2_thresh = np.percentile(cor_matrix, r2_thresh*100)
        
        if self.handle == 'mi_cor':
            # convert min_r2 to percentiles for mi
            min_r2 = np.percentile(self.rcf_series, min_r2*100)
        #'''
        selected_features = [index for index, value in self.rcf_series.items() if value > min_r2]
        i=0
        while np.any(cor_matrix>r2_thresh):
            row_bool = cor_matrix.iloc[i]>r2_thresh
            if np.any(row_bool):
                columns_to_remove = cor_matrix.iloc[i][row_bool].index.tolist()
                columns_to_remove.append(cor_matrix.index[i])
                max_feat = self.rcf_series[columns_to_remove].idxmax()
                columns_to_remove.remove(max_feat)

                # Remove columns from selected_features
                for column in columns_to_remove:
                    if column in selected_features:
                        selected_features.remove(column)

                # Update cor_matrix with selected features
                cor_matrix = cor_matrix.loc[selected_features, selected_features]
                i=0 #start from first row again
            else:
                i+=1
        self.selected_features = selected_features
        return selected_features