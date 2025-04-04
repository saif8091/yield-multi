from csv import excel
import pickle
import numpy as np
import xarray as xr
import rioxarray
import re
from src.prep_func import dn_sample_sig
from data_load.gt_data_load import *
from data_load.wt_data_load import *

# loading image data
prep_veg_h = pickle.load(open('data/preprocessed/prep_veg_mask_h.pkl', 'rb'))
prep_veg_h_21 = prep_veg_h['2021']
prep_veg_h_22 = prep_veg_h['2022']
bin_masks_21_h = prep_veg_h['2021_bin']
bin_masks_22_h = prep_veg_h['2022_bin']

veg_m = pickle.load(open('data/preprocessed/veg_m.pkl', 'rb'))
veg_m_21 = veg_m['2021']
veg_m_22 = veg_m['2022']

veg_chm = pickle.load(open('data/preprocessed/veg_chm.pkl', 'rb'))
veg_chm_21 = veg_chm['2021']
veg_chm_22 = veg_chm['2022']

veg_chm_lidar_21 = veg_chm['2021_lidar']
veg_chm_lidar_22 = veg_chm['2022_lidar']

veg_m_m = pickle.load(open('data/preprocessed/veg_m_m.pkl', 'rb'))
veg_m_21_m = veg_m_m['2021']
veg_m_22_m = veg_m_m['2022']
bin_masks_21_m = veg_m_m['2021_bin']
bin_masks_22_m = veg_m_m['2022_bin']

veg_chm_m = pickle.load(open('data/preprocessed/veg_chm_m.pkl', 'rb'))
veg_chm_21_m = veg_chm_m['2021']
veg_chm_22_m = veg_chm_m['2022']

veg_chm_lidar_21_m = veg_chm_m['2021_lidar']
veg_chm_lidar_22_m = veg_chm_m['2022_lidar']

lb, ub = 0, 80
#panels_h_21 = pickle.load(open('data/preprocessed/panels/panels_h_21.pkl', 'rb'))
hyper_wave = np.linspace(398.573,1001.81,272)#list(float(re.sub('[a-zA-Z]', '', element)) for element in panels_h_21['20210715']['NBL'].long_name)
hyper_wave_3 = dn_sample_sig(hyper_wave, 3)[lb:ub]
multi_wave = [475,560,668,717,840]