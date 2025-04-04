import os
import numpy as np
import rioxarray
from matplotlib import pyplot as plt
from tifffile import tifffile
from skimage import transform as transf

def zip_images(r_dir, panels=True, div_100=False):
    """
    This function takes a directory of images and loads them into a dictionary as xarray.
    The dictionary is structured with flight dates as keys, each containing another dictionary with plot numbers as keys and the corresponding image as the value.

    Parameters:
    r_dir (str): The directory where the images are stored.
    panels (bool): If True, the function assumes that the images are named with the plot number at the beginning of the filename. If False, the function assumes that the images are named with the plot number as strings.
    div_100 (bool): If True, the function divides the image values by 100.

    Returns:
    plot_im_all (dict): A dictionary containing the processed images, structured by flight date and plot number.
    Example usage: plot_im_all['20210707'][11] to access the image for flight date 20210707, plot 11.
    """
    plot_im_all = {}
    count = 0
    for flt_date in os.listdir(r_dir):
        flt_dir = os.path.join(r_dir, flt_date)
        if os.path.isdir(flt_dir):
            plot_im_all[flt_date] = {}
            for rfile in os.listdir(flt_dir):
                rpath = os.path.join(flt_dir, rfile)
                im = rioxarray.open_rasterio(rpath)
                if div_100:
                    im = im / 100
                if panels:
                    plt_num = rfile.split('_')[0] # plot number is the first part of the filename and to make this function usable for both panels and plots
                else:
                    plt_num = int(rfile.split('_')[0])
                plot_im_all[flt_date][plt_num] = im
                count += 1
    print(f'zipped {count} images')
    return plot_im_all

def zip_images22(r_dir, div_100=False):
    """
    This function takes a directory of images and loads them into a dictionary as xarray. Use  this for 2022 data.
    The dictionary is structured with flight dates as keys, each containing another dictionary with plot numbers as keys and the corresponding image as the value.

    Parameters:
    r_dir (str): The directory where the images are stored.
    div_100 (bool): If True, the function divides the image values by 100.
    
    Returns:
    plot_im_all (dict): A dictionary containing the processed images, structured by flight date and plot number.
    Example usage: plot_im_all['20210707'][11] to access the image for flight date 20210707, plot 11.
    """
    plot_im_all = {}
    count = 0
    for flt_date in os.listdir(r_dir):
        flt_dir = os.path.join(r_dir, flt_date)
        if os.path.isdir(flt_dir):
            plot_im_all[flt_date] = {}
            for grid in os.listdir(flt_dir):
                grid_dir = os.path.join(flt_dir, grid)
                if os.path.isdir(grid_dir):
                    for rfile in os.listdir(grid_dir):
                        rpath = os.path.join(grid_dir, rfile)
                        im = rioxarray.open_rasterio(rpath)
                        if div_100:
                            im = im / 100
                        plt_num = int(rfile.split('_')[0])
                        if grid == 'UV_efficacy_2022':
                            plt_num += 40
                        elif grid == 'lovebeets_grid_2022':
                            plt_num += 70
                        plot_im_all[flt_date][plt_num] = im
                        count += 1
    print(f'zipped {count} images')
    return plot_im_all