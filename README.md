<div align="center">

# Table Beets multiseason yield

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://python.org)

</div>

## **HSI Preprocessing**
* Spectral downsampling by 3 (mean of 3 adjacent bands)
* Cut off extremities
* Savitsky Golay filter

## **Feature names**
gdd: Growing degree days

evap: Accumulated evapotranspiration

vol: Volume (obtained from SFM)

vol_lidar: Volume (obtained from LiDAR)

The mean reflectance spectra of each plot follows the following naming convention" 'ref_x_mean_n'
here, x could be h or m corresponding to hyperspectral and multispectral respectively while n refers to the indices number. For HSI there are 80 indices so n can be 0-79, corresponding to the wavelength between 400.80 to 928.35 nm.

Spectral decompostion given by: xxx_h_n,
here xxx could be pca, fa, ica or plsr and n refers to the band number. I did 3 component decomposition cause 95% variance are explained by 3 components.

Finally the vegetation indices spectra are refered by: vi_xxx_h_mean
where xxx refers to the vegetation index, and again h means VI obtained from hyperspectral while m means its obtained from multispectral.

Note: Both spectral decompostion and vegetation indices were calculated from the mean plot reflectance spectra.

## Code
feat_split.py: Spliting the data set into test and train. This follows a split based on plot numbers.
run_gpr_with_feats.py: Runs multiple GPR with various feature combinations and records the performances.