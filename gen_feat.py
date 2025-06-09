import numpy as np
import pandas as pd
from tqdm import tqdm
from data_load.all_data_load import *
from src.feature_extraction import *

feat_file = 'data/preprocessed/features_21_22.csv'

def ratio_meteo_21_22(met_dict):
    """
    Ratio meteorological data by the harvest meteorological data.

    Returns:
    dict: A dictionary where the first key is the flight number and the second key is the plot number-1.
    """
    ratio_ = {}
    harvest_ = {}
    for date, value in met_dict.items():

        if str(date).startswith('2021'):
            ratio_[date] = np.ones((18,)) * met_dict[date]
            harvest_[date] = np.ones((18,)) * met_dict['20210805']
        elif str(date).startswith('2022'):
            total_plot = 88
            ratio_[date] = np.zeros((total_plot,))
            harvest_[date] = np.zeros((total_plot,))
            for i in range(total_plot):
                if i < 40:
                    ratio_[date][i] = met_dict[date]
                    harvest_[date][i] = met_dict['20220818']
                else:
                    ratio_[date][i] = met_dict[date] 
                    harvest_[date][i] = met_dict['20220824']
        else:
            raise ValueError('Date not found')

    return ratio_, harvest_

def zip_feat_(veg_chm, veg_m, veg_lidar, veg_lidar_m, veg_h, wave_comb_m, wave_comb_h, ratio_gdd_dict, ratio_evap_dict, target_variable):
    '''
    This function takes a dictionary of images and a dataframe of target variables
    Returns a dataframe containing the features and target variables
    '''
    feat=[]
    pbarflt = tqdm(total=len(veg_chm), desc=f'Total flights completed')
    for flight_date, flt_im in veg_chm.items():
        pbarplt = tqdm(total=len(flt_im), desc=f'Extracting features for {flight_date}')
        for plot_num, veg_im_chm in flt_im.items():
            feat.append({
            'Flight': flight_date,
            'Plot': plot_num,
            'gdd': ratio_gdd_dict[0][flight_date][plot_num-1],
            'gdd_harvest': ratio_gdd_dict[1][flight_date][plot_num-1],
            'evap': ratio_evap_dict[0][flight_date][plot_num-1],
            'evap_harvest': ratio_evap_dict[1][flight_date][plot_num-1],
            **features_from_single_veg_image(veg_im_chm, veg_m[flight_date][plot_num], veg_lidar[flight_date][plot_num], veg_lidar_m[flight_date][plot_num], veg_h[flight_date][plot_num], wave_comb_m, wave_comb_h),
            'y': target_variable[flight_date][plot_num-1] / 1.34 # average area of the plot
            })
            pbarplt.update(1)
        pbarplt.close()
        pbarflt.update(1)
    pbarflt.close()
    return pd.DataFrame(feat)
features = zip_feat_(veg_chm = {**veg_chm_21_m,**veg_chm_22_m}, 
                     veg_m = {**veg_m_21_m,**veg_m_22_m},
                     veg_lidar = {**veg_chm_lidar_21,**veg_chm_lidar_22},
                     veg_lidar_m = {**veg_chm_lidar_21_m,**veg_chm_lidar_22_m}, 
                     veg_h = {**prep_veg_h_21,**prep_veg_h_22}, 
                     wave_comb_m=multi_wave,  
                     wave_comb_h = hyper_wave_3, 
                     ratio_gdd_dict = ratio_meteo_21_22(dates_gdd_dict), 
                     ratio_evap_dict = ratio_meteo_21_22(dates_evap_dict), 
                     target_variable = rootwt_comb)
features.to_csv(feat_file, index=False)
print(f'Features saved in {feat_file}')
