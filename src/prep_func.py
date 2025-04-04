import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from skimage.util import view_as_windows
from tqdm import tqdm

def dn_sample_sig(a, n):
    '''
    This function downsamples a 1D array by averaging n channels together.
    If the size of the array is not a multiple of n, the excess entries are dropped.
    a is the array and n is the number of channels averaged.
    https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
    '''
    len_new = len(a) // n * n  # New length of the array
    a_new = a[:len_new]  # Trim the array to the new length
    return np.array(a_new).reshape(-1, n).mean(axis=1)

def downsample_dataset(dataset, factor):
    """
    This function downsamples an xarray Dataset along the band dimension.

    Parameters:
    dataset (xarray.Dataset): The dataset to downsample. Format: channels, rows, cols.
    factor (int): The downsampling factor.

    Returns:
    xarray.Dataset: The downsampled dataset.
    """
    return dataset.coarsen(band=factor, boundary='trim').mean()

def apply_savgol_filter(dataset, window_length, polyorder):
    """
    Apply a Savitzky-Golay filter to a 3D hyperspectral xarray Dataset.

    Parameters:
    dataset (xarray.Dataset): The hyperspectral dataset to filter. Must be a 3D dataset. (channels, rows, columns)
    window_length (int): The length of the filter window (i.e., the number of points used for the polynomial fit). Must be an odd integer.
    polyorder (int): The order of the polynomial used in the Savitzky-Golay filter.

    Returns:
    xarray.Dataset: The filtered dataset.
    """
    # Convert the Dataset to a numpy array and apply the filter
    filtered_data = savgol_filter(dataset.to_numpy(), window_length, polyorder, axis=0)

    # Convert the filtered data back to a Dataset
    return xr.DataArray(filtered_data, coords=dataset.coords, dims=dataset.dims)  # note when performing conversion of numpy to xarray do not use .to_dataset() as it shifts the x and y coordinates

def apply_gaussian_blur(dataset, sigma):
    """
    Apply a Gaussian blur to a 3D hyperspectral xarray Dataset.

    Parameters:
    dataset (xarray.Dataset): The hyperspectral dataset to blur. Must be a 3D dataset. (channels, rows, columns)
    sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
    xarray.Dataset: The blurred dataset.
    """
    # Apply the Gaussian filter to the entire 3D dataset
    blurred_data = gaussian_filter(dataset.values, sigma=(0, sigma, sigma))

    # Convert the blurred data back to a Dataset
    return xr.DataArray(blurred_data, coords=dataset.coords, dims=dataset.dims)

def slice_raster(image, start_band, end_band):
    """
    Slice the raster image within specified band range.

    Parameters:
    - image: The image containing the data.
    - start_band: The starting index of the band range.
    - end_band: The ending index of the band range (inclusive).

    Returns:
    - Sliced image based on the provided parameters.
    """
    return image.isel(band=slice(start_band, end_band))
    
def spatial_downsample(image, kernel_size, stride=1, overlap=False):
    """
    Spatially downsample an image by taking the mean within kernels.

    Parameters:
        image (ndarray): Input image of shape (height, width, channels).
        kernel_size (int): Size of the kernel.
        stride (int): The stride of the kernel. Only used if overlap is True.
        overlap (bool): Whether to use overlapping kernels.

    Returns:
        downsampled_image (ndarray): Downsampled image.
    """
    if overlap:
        # Use view_as_windows to create overlapping kernels
        window_shape = (kernel_size, kernel_size, image.shape[2])
        windows = view_as_windows(image, window_shape, step=stride)
        downsampled_image = np.mean(windows, axis=(2, 3, 4))
    else:
        height, width, channels = image.shape
        downsampled_height = height // kernel_size
        downsampled_width = width // kernel_size

        # Reshape the image into non-overlapping kernels
        reshaped_image = image[:downsampled_height * kernel_size, :downsampled_width * kernel_size, :]
        reshaped_image = reshaped_image.reshape(downsampled_height, kernel_size, downsampled_width, kernel_size, channels)

        # Calculate the mean within each kernel
        downsampled_image = np.mean(reshaped_image, axis=(1, 3))

    return downsampled_image

def apply_functions_to_images(image_dict, functions):
    """
    Apply a list of functions in sequence to each image in a nested dictionary.

    Parameters:
    image_dict (dict): The nested dictionary of images.
    functions (list): A list of tuples. Each tuple contains a function and a dictionary of arguments to pass to the function.

    Returns:
    dict: A new nested dictionary with the functions applied to each image.

    # Example usage:
    #functions = [(dn_sample_img, {'n': 3}),(apply_savgol_filter, { 'window_length': 5, 'polyorder': 3}),(apply_gaussian_blur, {'sigma': 0.8})]
    #blurred_images = apply_functions_to_images(zipped_im21, functions)
    """
    # Create a new dictionary to hold the processed images
    processed_images = {}

    # Calculate the total number of images for processing
    total_images = sum(len(outer_value) for outer_value in image_dict.values())
    pbar = tqdm(total=total_images, desc=f'Processing images')

    # Iterate over the outer dictionary
    for outer_key, outer_value in image_dict.items():
        processed_images[outer_key] = {}

        # Iterate over the inner dictionary
        for inner_key, image in outer_value.items():
            # Apply each function in sequence
            for function, args in functions:
                image = function(image, **args)

            # Store the processed image in the new dictionary
            processed_images[outer_key][inner_key] = image
            pbar.update(1)
    pbar.close()    
            
    return processed_images