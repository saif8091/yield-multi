import numpy as np
import pickle
from src.prep_func import *

# Load the zipped images
zipped_h = pickle.load(open('data/preprocessed/zipped_h.pkl', 'rb'))

zipped_im21 = zipped_h['2021']
zipped_im22 = zipped_h['2022']

# Create a list of functions to apply to the images
functions = [(downsample_dataset, {'factor': 3}),
             (apply_savgol_filter, {'window_length': 5, 'polyorder': 3}),
             (slice_raster, {'start_band': 0, 'end_band': 80})]
             #(apply_gaussian_blur, {'sigma': 0.8})]

# Apply the functions to the images
prep_21 = apply_functions_to_images(zipped_im21, functions)
prep_22 = apply_functions_to_images(zipped_im22, functions)

# Save the processed images
prep_h = {
    '2021': prep_21,
    '2022': prep_22
}
pickle.dump(prep_h, open('data/preprocessed/prep_h.pkl', 'wb'))
print('zipping images completed!!! Files saved as data/preprocessed/prep_h.pkl')