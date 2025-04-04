import numpy as np
import pandas as pd

# loading root weight data in kg
excel_21 = pd.read_excel('data/2021_data.xlsx', sheet_name='Data', header=5)
excel_22_disease = pd.read_excel('data/2022_data.xlsx', sheet_name='LBRN12DISEASE')
excel_22_uvc = pd.read_excel('data/2022_data.xlsx', sheet_name='LB RN12WEST')
excel_22_lovebeets = pd.read_excel('data/2022_data.xlsx', sheet_name='LB RN12EAST')

rootwt_21 = excel_21['Rootwt'][:18].values
rootwt_22 = pd.concat([excel_22_disease['Root weight (g)'], excel_22_uvc['Root weight (g)'], excel_22_lovebeets['Root weight (g)']]).values.astype(float)/1000

fol_21 = excel_21['DW foliage'][:18].values
fol_22 = pd.concat([excel_22_disease['Dry weight of foliage (g)'], excel_22_uvc['Dry weight of foliage (g)'], excel_22_lovebeets['Dry weight of foliage (g)']]).values

rootnum_21 = excel_21['rootnumber'][:18].values.astype(float)
rootnum_22 = pd.concat([excel_22_disease['Root number'], excel_22_uvc['Root number'], excel_22_lovebeets['Root number']]).values.astype(float)

diam_21 = excel_21['Average shoulderdiam'][:18].values
diam_22 = pd.concat([excel_22_disease['Average root diameter (mm)'], excel_22_uvc['Average root diameter (mm)'], excel_22_lovebeets['Average root diameter (mm)']]).values

def combine_rootwt(rootwt_21, rootwt_22, flights):
    """
    Combine two arrays into a dictionary based on a list of flights.

    Parameters:
    rootwt_21 (numpy.ndarray): The first array.
    rootwt_22 (numpy.ndarray): The second array.
    flights (list): The list of flights.

    Returns:
    dict: A dictionary where the first key is the flight number and the second key is the plot number-1.
    """
    combined_rootwt = {}

    for flight in flights:

        if str(flight).startswith('2021'):
            combined_rootwt[flight] = np.zeros(len(rootwt_21))
            for plot, value in enumerate(rootwt_21):
                combined_rootwt[flight][plot] = value
        elif str(flight).startswith('2022'):
            combined_rootwt[flight] = np.zeros(len(rootwt_22))
            for plot, value in enumerate(rootwt_22):
                combined_rootwt[flight][plot] = value

    return combined_rootwt

flight_dates = ['20210707', '20210715', '20210720', '20210802',
                '20220628', '20220707', '20220715', '20220726', '20220810', '20220818']

rootwt_comb = combine_rootwt(rootwt_21, rootwt_22, flight_dates)
fol_comb = combine_rootwt(fol_21, fol_22, flight_dates)
rootnum_comb = combine_rootwt(rootnum_21, rootnum_22, flight_dates)
diam_comb = combine_rootwt(diam_21, diam_22, flight_dates)