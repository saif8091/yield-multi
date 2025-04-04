import numpy as np
import rioxarray
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from skimage import filters, morphology # type: ignore
import copy

def threshold_ignore_nan(image, threhold_function):
    """
    Compute threshold for an image, ignoring NaN values.

    Parameters:
    image (numpy.ndarray): The input image as a numpy array.
    threhold_function (callable): The threshold function to use (e.g., filters.threshold)

    Returns:
    float: The computed threshold value.
    """
    # Flatten the array and remove NaN values
    cleaned_values = image.values.flatten()
    cleaned_values = cleaned_values[~np.isnan(cleaned_values)]

    # Apply threshold_otsu to the array without NaN values
    threshold_value = threhold_function(cleaned_values)
    return threshold_value

class BinaryMapGenerator:
    """
    A class to generate binary maps from hyperspectral/multispectral images or similar multidimensional data, highlighting specific features such as vegetation.
    
    This class processes images by computing a distance map, applying a threshold to create a binary map based on whether the distance is greater than or less than the threshold (configurable), optionally performing a morphological operation to refine the map, and extracting the feature of interest from the original image using the refined binary map.
    
    Attributes:
        distance_function (callable): A function to compute the distance map. It must accept an image and optionally additional parameters.
        distance_function_params (dict, optional): Parameters for the distance function.
        threshold_function (callable): A function to compute the threshold value for binarization. Default is Otsu's method.
        threshold_direction (str): Direction for threshold comparison, 'greater' for values greater than the threshold or 'less' for values less than the threshold. Default is 'greater'.
        black_region_width (int): Width of the region to be set to black (0) on both sides of the binary map.
        morph_func (callable): Morphological operation function to apply on the binary map. Default is opening.
        morph_func_params (dict, optional): Parameters for the morphological operation function.
    
    Methods:
        _calculate_distance_map(im): Calculates the distance map of the image.
        _calculate_threshold(im): Calculates the threshold value for binarization.
        _create_binary_map(im): Creates a binary map using the threshold value.
        _apply_generate_mask(im): Applies the specified morphological operation to refine the binary map.
        _extract_vegetation_image(im): Extracts the feature of interest from the original image using the refined binary map.
        _plot_results(): Plots the results of the binary map generation process.
    Examples:
        >>> from skimage import filters, morphology
        >>> bin_map_gen_class = BinaryMapGenerator(
        ...     distance_function=calculate_distance,
        ...     distance_function_params={'spec': veg_spec, 'distance_func': SAM},
        ...     threshold_function=filters.threshold_otsu, 
        ...     threshold_direction = 'less', 
        ...     black_region_width=3, 
        ...     morph_func=morphology.opening,
        ...     morph_func_params={'footprint': morphology.disk(3)}
        ... )
        >>> bin_map_gen_class._plot_results(prep_h_22['20220810'][41])
    """
    def __init__(self, distance_function, distance_function_params=None, threshold_function=filters.threshold_otsu, threshold_direction='greater', black_region_width=3, morph_func=morphology.opening, morph_func_params=None):
        self.distance_function = distance_function
        self.distance_function_params = distance_function_params if distance_function_params is not None else {}
        self.threshold_function = threshold_function
        self.threshold_direction = threshold_direction
        self.black_region_width = black_region_width
        self.morph_func = morph_func
        self.morph_func_params = morph_func_params if morph_func_params is not None else {}
        # Initialize variables to store intermediate results
        self.im = None
        self.distance_map = None
        self.threshold_value = None
        self.map_binary = None
        self.map_binary_morph = None
        self.veg_im = None

    def _calculate_distance_map(self, im):
        """Calculates the distance map of the image."""
        if not isinstance(im, xr.DataArray):
            raise ValueError("Invalid input types for image use xarray DataArray.")
        self.im = im
        self.distance_map = self.distance_function(im, **self.distance_function_params)
        return self.distance_map

    def _calculate_threshold(self, im):
        """Calculates the threshold value for binarization."""
        self._calculate_distance_map(im)
        self.threshold_value = threshold_ignore_nan(self.distance_map, self.threshold_function)
        return self.threshold_value
    
    def _create_binary_map(self, im):
        """Creates a binary map using the threshold value and direction."""
        self._calculate_threshold(im)
        if self.distance_map is None or self.threshold_value is None:
            raise ValueError("Distance map or threshold value is not calculated yet.")
        if self.threshold_direction == 'greater':
            self.map_binary = self.distance_map > self.threshold_value
        else:  # 'less'
            self.map_binary = self.distance_map < self.threshold_value
        return self.map_binary
    
    def _generate_mask(self,im):
        """Applies the specified morphological operation to refine the binary map."""
        self._create_binary_map(im)
        if self.map_binary is None:
            raise ValueError("Binary map is not created yet.")
        binary_array = self.map_binary.values
        binary_array[:, :self.black_region_width] = 0
        binary_array[:, -self.black_region_width:] = 0
        binary_array = self.morph_func(binary_array, **self.morph_func_params)
        self.map_binary_morph = xr.DataArray(binary_array, dims=('y', 'x'), coords={'y': self.map_binary.coords['y'], 'x': self.map_binary.coords['x']})
        return self.map_binary_morph

    def _extract_vegetation_image(self, im):
        """Extracts the feature of interest from the original image using the refined binary map."""
        self._generate_mask(im)
        self.veg_im = im.where(self.map_binary_morph, drop=False)
        return self.veg_im
    
    '''
    def _plot_results(self, im, hyper=True):
        """Plots the results of the binary map generation process."""
        self._extract_vegetation_image(im)
        if self.im is None or self.distance_map is None or self.threshold_value is None or self.map_binary is None or self.map_binary_morph is None or self.veg_im is None:
            raise ValueError("One or more intermediate results are not calculated yet.")
        fig, axs = plt.subplots(2, 3, figsize=(12, 10))
        
        # Plot the main image
        if hyper:
            self.im[[36,23,11]].plot.imshow(ax=axs[0, 0], robust=True)
        else:
            self.im[[2,1,0]].plot.imshow(ax=axs[0, 0], robust=True)
        axs[0, 0].set_title('Main Image')
        
        # Plot the distance map
        self.distance_map.plot.imshow(ax=axs[0, 1])
        axs[0, 1].set_title('Distance Map')
        
        # Plot histogram of the distance map
        sns.histplot(self.distance_map.values.flatten(), kde=True, ax=axs[0, 2])
        axs[0, 2].set_title('Distance Map Histogram')
        plt.sca(axs[0, 2])
        plt.axvline(x=self.threshold_value, color='r', linestyle='--')
        plt.text(self.threshold_value, -0.5, f'{self.threshold_value:.3f}', color='r', va='top')
        
        # Plot binary map
        self.map_binary.plot.imshow(ax=axs[1, 0])
        axs[1, 0].set_title('Binary Map')
        
        # Plot morphologically opened binary map
        self.map_binary_morph.plot.imshow(ax=axs[1, 1])
        axs[1, 1].set_title('Morphologically Opened Binary Map')
        
        # Plot the extracted veg image
        if hyper:
            self.veg_im[[36,23,11]].plot.imshow(ax=axs[1, 2], robust=True)
        else:
            self.veg_im[[2,1,0]].plot.imshow(ax=axs[1, 2], robust=True)
        axs[1, 2].set_title('Extracted Image')
        
        plt.tight_layout()
        plt.show()
'''

def apply_mask_to_image(xarray_image, binary_mask, chm_correction=False):
    """
    Applies a binary mask to an xarray image, masking out areas not covered by the mask.

    This function reindexes the given binary mask to match the coordinates of the xarray image. It fills any missing values in the mask with 0 (interpreted as False), ensuring the mask is boolean. The mask is then applied to the image, setting pixels outside the mask to NaN, effectively masking them out.

    Parameters:
    - xarray_image (xarray.DataArray): The input image as an xarray DataArray on which the mask will be applied.
    - binary_mask (xarray.DataArray): The binary mask as an xarray DataArray that defines the areas to keep in the xarray_image.

    Returns:
    - masked_image (xarray.DataArray): The resulting image after applying the mask, with areas outside the mask set to NaN.
    """
    # Reindex mask to match image coordinates, filling missing values with 0 (False)
    reindexed_mask = binary_mask.reindex_like(xarray_image, method='nearest', tolerance=1e-1).fillna(0).astype(bool)

    # Apply reindexed mask to image
    masked_image = xarray_image.where(reindexed_mask, np.nan)

    if chm_correction:
        # Calculate the average of the masked-out values
        average_masked_out = xarray_image.where(~reindexed_mask).mean().item()
        masked_image = masked_image-average_masked_out
        
    return masked_image

def veg_extract_using_masks(chm_dict, bin_dict, chm_correction=False):
    '''
    Extracts vegetation from a dictionary of chm or image dict using binary mask.
    '''
    # Create dictionaries to hold the processed and binary images
    processed_chm = {}

    # Iterate over the outer dictionary
    for outer_key, outer_value in bin_dict.items():
        processed_chm[outer_key] = {}

        # Iterate over the inner dictionary
        for inner_key, bin_image in outer_value.items():
            if outer_key not in chm_dict:
                continue
            processed_chm[outer_key][inner_key] = apply_mask_to_image(chm_dict[outer_key][inner_key], bin_image,chm_correction=chm_correction)

    return processed_chm

import copy

def apply_and_on_flights(bin_dict, bin_flight, flights, plots):
    """
    Applies a binary mask to each image in a vegetation dictionary for a specific flight, updating both the vegetation and binary mask dictionaries.

    This function iterates over each plot in the specified flight within the vegetation dictionary. For each plot, it applies a binary mask from the binary mask dictionary to the corresponding vegetation image. The binary mask is also reindexed to match the dimensions of the vegetation image, ensuring alignment. After applying the mask, the vegetation dictionary is updated with the masked images, and the binary mask dictionary is updated with the reindexed masks.

    Parameters:
    - bin_dict (dict): A dictionary containing binary masks for the vegetation images, structured as {flight: {plot: image}}.
    - flight (str): The key in bin_dict indicating the specific flight to process.
    - bin_flight (str): The key in bin_dict indicating the specific flight's binary masks to use for masking the vegetation images.

    Returns:
    - tuple: A tuple containing the updated veg_dict and bin_dict after applying the binary masks to the vegetation images.
    """
    # Create deep copies of the dictionaries to work on
    #veg_dict_copy = copy.deepcopy(veg_dict)
    bin_dict_copy = copy.deepcopy(bin_dict)

    for flight in flights:
        for plot in plots:
            bin_dict_copy[flight][plot] = bin_dict[flight][plot] & bin_dict_copy[bin_flight][plot].reindex_like(bin_dict[flight][plot], method='nearest', tolerance=1e-1).fillna(0).astype(bool) 
    
    return bin_dict_copy

def copy_flight_plot(bin_dict, bin_flight, flights, plots):
    """
    Copies binary mask for a specific flight and plot.

    This function iterates over each plot in the specified flight within the vegetation dictionary. For each plot, 
    it applies a binary mask from the binary mask dictionary to the corresponding vegetation image. The binary mask 
    is also reindexed to match the dimensions of the vegetation image, ensuring alignment. After applying the mask, 
    the vegetation dictionary is updated with the masked images, and the binary mask dictionary is updated with the reindexed masks.

    Parameters:
    - bin_dict (dict): A dictionary containing binary masks for the vegetation images, structured as {flight: {plot: image}}.
    - flight (str): The key in bin_dict indicating the specific flight to process.
    - bin_flight (str): The key in bin_dict indicating the specific flight's binary masks to use for masking the vegetation images.

    Returns:
    - tuple: A tuple containing the updated veg_dict and bin_dict after applying the binary masks to the vegetation images.
    """
    # Create deep copies of the dictionaries to work on
    bin_dict_copy = copy.deepcopy(bin_dict)

    for flight in flights:
        for plot in plots:
            bin_dict_copy[flight][plot] = bin_dict[bin_flight][plot].reindex_like(bin_dict[flight][plot], method='nearest', tolerance=1e-1).fillna(0).astype(bool) 
    
    return bin_dict_copy