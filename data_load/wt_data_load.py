import numpy as np
import pandas as pd
from datetime import datetime

def calculate_weather_metrics(data, start_date, end_date, t_base, evap=True):
    """
    Calculate weather metrics including Growing Degree Days (GDD) and cumulative evaporation.
    
    Parameters:
    - data: Path to the CSV file containing weather data.
    - start_date: The start date for the period of interest (inclusive).
    - end_date: The end date for the period of interest (inclusive).
    - t_base: The base temperature for calculating GDD, below which GDD is considered to be zero.
    
    Returns:
    - cumulative_gdd: A Series representing the cumulative Growing Degree Days over the specified period.
    - cumulative_evap: A Series representing the cumulative evaporation over the specified period.
    """
    # Load the weather data    
    data = pd.read_csv(data, parse_dates=['DATE'])
    # Filter data for the given date range
    filtered_data = data.loc[(data['DATE'] >= start_date) & (data['DATE'] <= end_date)]
    
    # Calculate Growing Degree Days (GDD)
    gdd = ((filtered_data['TMAX'] + filtered_data['TMIN']) / 2 - t_base)#.clip(lower=0)
    cumulative_gdd = np.cumsum(gdd).reset_index(drop=True).round(2) # type: ignore
    
    if not evap:
        return cumulative_gdd
    
    # Calculate Evaporation (EVAP)
    cumulative_evap = np.cumsum(filtered_data['EVAP'].fillna(0)).reset_index(drop=True).round(2) # type: ignore , fill the missing values with 0
    
    return cumulative_gdd, cumulative_evap

def weather_data_date(planting_date, analysis_date, data='data/all_weather_data.csv', t_base=4, evap=True):
    ''' 
    Function to calculate GDD and EVAP at the specific date.
    '''
    # Get the index for the analysis date
    day_index = (analysis_date - planting_date).days
    
    if not evap:
        # Calculate cumulative GDD
        cumulative_gdd = calculate_weather_metrics(data, planting_date, analysis_date, t_base, evap=False)
        gdd_value = cumulative_gdd.iloc[day_index] # type: ignore
        return gdd_value

    # Calculate cumulative GDD and EVAP
    cumulative_gdd, cumulative_evap = calculate_weather_metrics(data, planting_date, analysis_date, t_base, evap=evap)
    
    # Retrieve GDD and EVAP for the specific day
    gdd_value = cumulative_gdd.iloc[day_index]
    evap_value = cumulative_evap.iloc[day_index]
    
    return gdd_value, evap_value

#weather = pd.read_csv('data/all_weather_data.csv')

gdd_20210616, evap_20210616 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-06-16'))
gdd_20210707, evap_20210707 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-07-07'))
gdd_20210715, evap_20210715 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-07-15'))
gdd_20210720, evap_20210720 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-07-20'))
gdd_20210802, evap_20210802 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-08-02'))
gdd_20210805, evap_20210805 = weather_data_date(pd.to_datetime('2021-05-20'), pd.to_datetime('2021-08-05'))

gdd_20220610, evap_20220610 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-06-10'))
gdd_20220628, evap_20220628 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-06-28'))
gdd_20220707, evap_20220707 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-07-07'))
gdd_20220715, evap_20220715 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-07-15'))
gdd_20220726, evap_20220726 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-07-26'))
gdd_20220810, evap_20220810 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-08-10'))
gdd_20220818, evap_20220818 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-08-18'))
gdd_20220824, evap_20220824 = weather_data_date(pd.to_datetime('2022-05-18'), pd.to_datetime('2022-08-24'))

dates = ['20210707', '20210715', '20210720', '20210802', '20210805',
           '20220628', '20220707', '20220715', '20220726', '20220810', '20220818', '20220824']
gdd = [gdd_20210707, gdd_20210715, gdd_20210720, gdd_20210802, gdd_20210805,
       gdd_20220628, gdd_20220707, gdd_20220715, gdd_20220726, gdd_20220810, gdd_20220818, gdd_20220824]

evap = [evap_20210707, evap_20210715, evap_20210720, evap_20210802, evap_20210805,
       evap_20220628, evap_20220707, evap_20220715, evap_20220726, evap_20220810, evap_20220818, evap_20220824]

# Combine the two lists into a dictionary
dates_gdd_dict = dict(zip(dates, gdd))
dates_evap_dict = dict(zip(dates, evap))