<div align="center">

# Multistage Table Beets yield prediction from unmanned aerial systems

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)

</div>

# **Incomplete Repository!!!**

## 🔄**Preprocessing and Vegetation extraction**
Run the following code:
```shell
python make.py
```
This code zips the images, performs preprocessing (HSI only) and then extracts the vegetation from the croped plot images.

### **HSI Preprocessing**
* Spectral downsampling by averaging every 3 adjacent bands
* Savitsky Golay filter
* Cut off extremities

## **Feature generation**
```shell
python gen_feat.py
```
This code generates different types of features of each plot and compiles them in a csv file and can be found [here](data/preprocessed/features_21_22.csv).

### Feature Abbreviations and Definitions

- **gdd**: Growing Degree Days
- **evap**: Accumulated Evapotranspiration
- **vol**: Volume (obtained from SFM)
- **vol_lidar**: Volume (obtained from LiDAR)
- **y**: Beet root yield ($kg/m^2$)

### Naming Conventions

#### Mean Reflectance Spectra
- **Format**: `ref_x_mean_n`
  - `x`: `h` for hyperspectral, `m` for multispectral
  - `n`: Index number (0-79 for HSI, corresponding to wavelengths between 400.80 to 928.35 nm)

#### Spectral Decomposition
- **Format**: `xxx_h_n`
  - `xxx`: Decomposition method (`pca`, `fa`, `ica`, `plsr`)
  - `h`: Indicates hyperspectral data
  - `n`: Band number
  - Note: 3-component decomposition is used as it explains 95% of the variance.

#### Vegetation Indices Spectra
- **Format**: `vi_xxx_h_mean`
  - `xxx`: Vegetation index
  - `h`: Indicates hyperspectral data, `m` for multispectral data

### Notes
- Both spectral decomposition and vegetation indices are calculated from the mean plot reflectance spectra.

### **Feature filtering**
```shell
python filter.py
```
This is code is run to find the relevant spectral features. The filtered feature can be found [here.](feature_filter/filtered_features)

## **GPR Model Test**
```shell
python gpr_score_22.py
```
Trains and ouputs the GPR model scores at different feature combinations for the 2022 data set only. The scores can be found [here.](model_files/model_scores)

To test for transferability to 2021 test set run the following code:
```shell
python -m model_test_21.gpr_testing_21
```
To visualise the result open this [notebook.](model_test_21/visualising_performance.ipynb)
## Schematic of the Model
<p align="center">
  <img src="figures/model_schematic.png" alt="Schematic">
</p>