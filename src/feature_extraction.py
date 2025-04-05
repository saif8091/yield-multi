''' This file contains the functions for feature extraction from preprocessed images'''
from weakref import ref
import numpy as np
from src.veg_indices import *
import xarray as xr
import rioxarray
import re
from src.spec_decomp_func import *
from joblib import load

def total_area(xarr):
    '''Calculates the total plot area. Use CHMs only'''
    return np.size(xarr) * np.abs(xarr.rio.resolution()[0] * xarr.rio.resolution()[1])

def volume(xarr, **kwargs):
    '''Calculates the volume of the vegetation. Use CHMs only'''
    return np.nansum(xarr, **kwargs)*np.abs(xarr.rio.resolution()[0] * xarr.rio.resolution()[1])

def features_from_single_veg_image(veg_im_chm, veg_im_m, 
                                   veg_im_lidar, veg_im_lidar_m, 
                                   veg_im_h, wave_comb_m, wave_comb_h):
    '''
    This function calculates the features from a single masked images (non vegetation pixels are nan)
    Here, wave_comb is the wavelength combination for the image and veg_image is the masked image
    '''
    ## Mean Height
    mean_height = np.nanmean(veg_im_chm)
    mean_height_lidar = np.nanmean(veg_im_lidar)
    mean_height_lidar_m = np.nanmean(veg_im_lidar_m)

    ## Volume calculation
    veg_volume = volume(veg_im_chm) / total_area(veg_im_chm)
    veg_volume_lidar = volume(veg_im_lidar) / total_area(veg_im_lidar)
    veg_volume_lidar_m = volume(veg_im_lidar_m) / total_area(veg_im_lidar_m)
    
    ## Decompostion
    # loading decomposer models
    PCA_decomposer = load('data/preprocessed/decomposer/PCA_decomposer.joblib')
    FA_decomposer = load('data/preprocessed/decomposer/FA_decomposer.joblib')
    ICA_decomposer = load('data/preprocessed/decomposer/ICA_decomposer.joblib')
    PLSR_decomposer = load('data/preprocessed/decomposer/PLSR_decomposer.joblib')

    ## mean reflectance calculation
    ref_h_mean = np.nanmean(veg_im_h, axis=(1,2))
    ref_m_mean = np.nanmean(veg_im_m, axis=(1,2))

    pca_mean = PCA_decomposer.transform(ref_h_mean).flatten()
    fa_mean = FA_decomposer.transform(ref_h_mean).flatten()
    ica_mean = ICA_decomposer.transform(ref_h_mean).flatten()
    plsr_mean = PLSR_decomposer.transform(ref_h_mean).flatten()

    ## Vegetation indices
    ndvi_m = VI(wave_comb_m,840,668).ND(ref_m_mean)
    ndvi_h = VI(wave_comb_h,840,668).ND(ref_h_mean)

    rdvi_m = VI(wave_comb_m,840,668).RD(ref_m_mean)
    rdvi_h = VI(wave_comb_h,840,668).RD(ref_h_mean)

    gndvi_m = VI(wave_comb_m,800,570).ND(ref_m_mean)
    gndvi_h = VI(wave_comb_h,800,570).ND(ref_h_mean)

    ngrdi_m = VI(wave_comb_m,560,680).ND(ref_m_mean)
    ngrdi_h = VI(wave_comb_h,560,680).ND(ref_h_mean)

    wdrvi_m = VI(wave_comb_m,800,670).WDRVI(ref_m_mean)
    wdrvi_h = VI(wave_comb_h,800,670).WDRVI(ref_h_mean)

    pbi_m = VI(wave_comb_m,810,560).ratio(ref_m_mean)
    pbi_h = VI(wave_comb_h,810,560).ratio(ref_h_mean)

    lci_m = VI(wave_comb_m,850,710,670).RD3(ref_m_mean)
    lci_h = VI(wave_comb_h,850,710,670).RD3(ref_h_mean)

    psnd_m = VI(wave_comb_m,800,470).ND(ref_m_mean)
    psnd_h = VI(wave_comb_h,800,470).ND(ref_h_mean)

    msr_m = VI(wave_comb_m,800,760,670).MSR(ref_m_mean)
    msr_h = VI(wave_comb_h,800,760,670).MSR(ref_h_mean)

    psri_m = VI(wave_comb_m,680,500,750).PSR(ref_m_mean)
    psri_h = VI(wave_comb_h,680,500,750).PSR(ref_h_mean)

    npci_m = VI(wave_comb_m,670,460).ND(ref_m_mean)
    npci_h = VI(wave_comb_h,670,460).ND(ref_h_mean)

    osavi_m = VI(wave_comb_m,800,670).SA(ref_m_mean,l=0.16)
    osavi_h = VI(wave_comb_h,800,670).SA(ref_h_mean,l=0.16)

    evi2_m = VI(wave_comb_m,800,670).EVI2(ref_m_mean)
    evi2_h = VI(wave_comb_h,800,670).EVI2(ref_h_mean)

    tcari_m = VI(wave_comb_m,700,670,550).TCARI(ref_m_mean)
    tcari_h = VI(wave_comb_h,700,670,550).TCARI(ref_h_mean)

    spvi_m = VI(wave_comb_m,800,670,530).SPVI(ref_m_mean)
    spvi_h = VI(wave_comb_h,800,670,530).SPVI(ref_h_mean)

    tvi_m = VI(wave_comb_m,750,670,550).TVI(ref_m_mean)
    tvi_h = VI(wave_comb_h,750,670,550).TVI(ref_h_mean)

    mcari_m = VI(wave_comb_m,700,670,550).MCARI(ref_m_mean)
    mcari_h = VI(wave_comb_h,700,670,550).MCARI(ref_h_mean)

    gi_m = VI(wave_comb_m,554,677).ratio(ref_m_mean)
    gi_h = VI(wave_comb_h,554,677).ratio(ref_h_mean)

    ### Only HSI VI
    ci_red_edge_h = VI(wave_comb_h,780,710).ratio(ref_h_mean)-1
    ci_chloro_h = VI(wave_comb_h,780,550).ratio(ref_h_mean)-1
    ndre_h = VI(wave_comb_h,790,720).ND(ref_h_mean)
    nd705_h = VI(wave_comb_h,750,705).ND(ref_h_mean)
    pssr_h = VI(wave_comb_h,800,500).ratio(ref_h_mean)
    rars_h = VI(wave_comb_h,760,500).ratio(ref_h_mean)
    sr_h = VI(wave_comb_h,750,550).ratio(ref_h_mean)
    rvsi_h = VI(wave_comb_h,752,732,712).RVSI(ref_h_mean)
    nd_560_620 = VI(wave_comb_h,560,620).ND(ref_h_mean)
    ratio_560_620 = VI(wave_comb_h,560,620).ratio(ref_h_mean)

    return  {
            **{f'mean_height': mean_height},
            **{f'mean_height_lidar': mean_height_lidar},
            **{f'mean_height_lidar_m': mean_height_lidar_m},
            **{f'vol': veg_volume},
            **{f'vol_lidar': veg_volume_lidar},
            **{f'vol_lidar_m': veg_volume_lidar_m},
            **{f'vi_ndvi_m_mean': ndvi_m},
            **{f'vi_ndvi_h_mean': ndvi_h},
            **{f'vi_rdvi_m_mean': rdvi_m},
            **{f'vi_rdvi_h_mean': rdvi_h},
            **{f'vi_gndvi_m_mean': gndvi_m},
            **{f'vi_gndvi_h_mean': gndvi_h},
            **{f'vi_ngrdi_m_mean': ngrdi_m},
            **{f'vi_ngrdi_h_mean': ngrdi_h},
            **{f'vi_wdrvi_m_mean': wdrvi_m},
            **{f'vi_wdrvi_h_mean': wdrvi_h},
            **{f'vi_pbi_m_mean': pbi_m},
            **{f'vi_pbi_h_mean': pbi_h},
            **{f'vi_lci_m_mean': lci_m},
            **{f'vi_lci_h_mean': lci_h},
            **{f'vi_psnd_m_mean': psnd_m},
            **{f'vi_psnd_h_mean': psnd_h},
            **{f'vi_msr_m_mean': msr_m},
            **{f'vi_msr_h_mean': msr_h},
            **{f'vi_psri_m_mean': psri_m},
            **{f'vi_psri_h_mean': psri_h},
            **{f'vi_npci_m_mean': npci_m},
            **{f'vi_npci_h_mean': npci_h},
            **{f'vi_osavi_m_mean': osavi_m},
            **{f'vi_osavi_h_mean': osavi_h},
            **{f'vi_evi2_m_mean': evi2_m},
            **{f'vi_evi2_h_mean': evi2_h},
            **{f'vi_tcari_m_mean': tcari_m},
            **{f'vi_tcari_h_mean': tcari_h},
            **{f'vi_spvi_m_mean': spvi_m},
            **{f'vi_spvi_h_mean': spvi_h},
            **{f'vi_tvi_m_mean': tvi_m},
            **{f'vi_tvi_h_mean': tvi_h},
            **{f'vi_mcari_m_mean': mcari_m},
            **{f'vi_mcari_h_mean': mcari_h},
            **{f'vi_gi_m_mean': gi_m},
            **{f'vi_gi_h_mean': gi_h},
            **{f'vi_ci_red_edge_h_mean': ci_red_edge_h},
            **{f'vi_ci_chloro_h_mean': ci_chloro_h},
            **{f'vi_ndre_h_mean': ndre_h},
            **{f'vi_nd705_h_mean': nd705_h},
            **{f'vi_pssr_h_mean': pssr_h},
            **{f'vi_rars_h_mean': rars_h},
            **{f'vi_sr_h_mean': sr_h},
            **{f'vi_rvsi_h_mean': rvsi_h},
            **{f'vi_nd_560_620_h_mean': nd_560_620},
            **{f'vi_ratio_560_620_h_mean': ratio_560_620},
            **{f'pca_h_{index}': value for index, value in enumerate(pca_mean)},
            **{f'fa_h_{index}': value for index, value in enumerate(fa_mean)},
            **{f'ica_h_{index}': value for index, value in enumerate(ica_mean)},
            **{f'plsr_h_{index}': value for index, value in enumerate(plsr_mean)},
            **{f'ref_m_mean_{index}': value for index, value in enumerate(ref_m_mean)},
            **{f'ref_h_mean_{index}': value for index, value in enumerate(ref_h_mean)}
        }
