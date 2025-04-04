import numpy as np
import pickle
from src.misc import *
from src.prep_func import *
from src.veg_extract_func import *
from src.veg_indices import VI
import xarray as xr
from skimage import morphology, filters # type: ignore
import copy

#Load preprocessed images
prep_h = pickle.load(open('data/preprocessed/prep_h.pkl', 'rb'))
prep_h_21 = prep_h['2021']
del prep_h_21['20210616'] # remove the 20210616 flight date
prep_h_22 = prep_h['2022']

dict_m = pickle.load(open('data/preprocessed/zipped_m.pkl', 'rb'))
dict_m_21 = dict_m['2021']
dict_m_22 = dict_m['2022']

dict_chm = pickle.load(open('data/preprocessed/zipped_chm.pkl', 'rb'))
dict_chm_21 = dict_chm['2021']
dict_chm_22 = dict_chm['2022']

dict_chm_lidar_21 = dict_chm['2021_lidar']
dict_chm_lidar_22 = dict_chm['2022_lidar']

#Load wavelength bands
lb, ub = 0, 80
hyper_wave = np.linspace(398.573,1001.81,272)#list(float(re.sub('[a-zA-Z]', '', element)) for element in panels_h_21['20210715']['NBL'].long_name)
hyper_wave_3 = dn_sample_sig(hyper_wave, 3)[lb:ub]
multi_wave = [475,560,668,717,840]

# Generate vegetation binary map using RDVI
bin_map_gen = BinaryMapGenerator(
    distance_function = VI(hyper_wave_3,840,668).RD,
    threshold_function=filters.threshold_otsu, 
    threshold_direction = 'greater', 
    black_region_width=3, 
    morph_func=morphology.opening,
    morph_func_params={'footprint': morphology.disk(2)}
)
functions = [(bin_map_gen._generate_mask, {})]
bin_masks_21 = apply_functions_to_images(prep_h_21, functions)
bin_masks_22 = apply_functions_to_images(prep_h_22, functions)

# Code for restricting the binary masks to remove weeds
flights = ['20220810', '20220818']
plots = [11, 19, 31, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         55, 56, 57, 58, 78, 83, 84, 86, 87]
bin_masks_22 = apply_and_on_flights(bin_masks_22, '20220726', flights, plots)

# Extract vegetation using binary masks
veg_h_21 = veg_extract_using_masks(prep_h_21, bin_masks_21)
veg_h_22 = veg_extract_using_masks(prep_h_22, bin_masks_22)

veg_m_21 = veg_extract_using_masks(dict_m_21, bin_masks_21)
veg_m_22 = veg_extract_using_masks(dict_m_22, bin_masks_22)

veg_chm_21 = veg_extract_using_masks(dict_chm_21, bin_masks_21,chm_correction=False)
veg_chm_22 = veg_extract_using_masks(dict_chm_22, bin_masks_22,chm_correction=False)

veg_chm_lidar_21 = veg_extract_using_masks(dict_chm_lidar_21, bin_masks_21,chm_correction=False)
veg_chm_lidar_22 = veg_extract_using_masks(dict_chm_lidar_22, bin_masks_22,chm_correction=False)

# Save the processed images
prep_veg_mask_h = {
    '2021': veg_h_21,
    '2022': veg_h_22,
    '2021_bin': bin_masks_21,
    '2022_bin': bin_masks_22
}

veg_m = {
    '2021': veg_m_21,
    '2022': veg_m_22,
}

veg_chm = {
    '2021': veg_chm_21,
    '2022': veg_chm_22,

    '2021_lidar': veg_chm_lidar_21,
    '2022_lidar': veg_chm_lidar_22
}
pickle.dump(prep_veg_mask_h, open('data/preprocessed/prep_veg_mask_h.pkl', 'wb'))
pickle.dump(veg_m, open('data/preprocessed/veg_m.pkl', 'wb'))
pickle.dump(veg_chm, open('data/preprocessed/veg_chm.pkl', 'wb'))

## Vegetation extraction using multispectral images
bin_map_gen_multi = BinaryMapGenerator(
    distance_function = VI(multi_wave,840,668).RD,
    threshold_function=filters.threshold_otsu, 
    threshold_direction = 'greater', 
    black_region_width=9, 
    morph_func=morphology.closing,
    morph_func_params={'footprint': morphology.disk(2)}
)
functions = [(bin_map_gen_multi._generate_mask, {})]
bin_maps_m_21 = apply_functions_to_images(dict_m_21, functions)
bin_maps_m_22 = apply_functions_to_images(dict_m_22, functions)
del bin_maps_m_21['20210616']
del bin_maps_m_21['20210825']
del bin_maps_m_22['20220610']

# Code for restricting the binary masks to remove weeds
flights = ['20220810', '20220818']
plots = [11, 19, 31, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         55, 56, 57, 58, 78, 83, 84, 86, 87]
bin_maps_m_22 = apply_and_on_flights(bin_maps_m_22, '20220726', flights, plots)

veg_m_21_m = veg_extract_using_masks(dict_m_21, bin_maps_m_21)
veg_m_22_m = veg_extract_using_masks(dict_m_22, bin_maps_m_22)
veg_chm_21_m = veg_extract_using_masks(dict_chm_21, bin_maps_m_21, chm_correction=False)
veg_chm_22_m = veg_extract_using_masks(dict_chm_22, bin_maps_m_22, chm_correction=False)

veg_chm_lidar_21_m = veg_extract_using_masks(dict_chm_lidar_21, bin_maps_m_21, chm_correction=False)
veg_chm_lidar_22_m = veg_extract_using_masks(dict_chm_lidar_22, bin_maps_m_22, chm_correction=False)

veg_m_m = {
    '2021': veg_m_21_m,
    '2022': veg_m_22_m,
    '2021_bin': bin_maps_m_21,
    '2022_bin': bin_maps_m_22
}
veg_chm_m = {
    '2021': veg_chm_21_m,
    '2022': veg_chm_22_m,
    
    '2021_lidar': veg_chm_lidar_21_m,
    '2022_lidar': veg_chm_lidar_22_m
}
pickle.dump(veg_m_m, open('data/preprocessed/veg_m_m.pkl', 'wb'))
pickle.dump(veg_chm_m, open('data/preprocessed/veg_chm_m.pkl', 'wb'))
print('Vegetation extraction completed!!! Vegetation and binary masks saved as data/preprocessed/prep_veg_mask_h.pkl, multispec veg as data/preprocessed/veg_m.pkl and while CHM saved as data/preprocessed/veg_chm.pkl')