# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa
"""

import csv

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import cos, sin, atan, atan2, radians, degrees

""" Input parameters"""

tz = 'Europe/Copenhagen'
latitude = 55.696166
longitude = 12.105216
name = 'Risoe'
axis_tilt=0
axis_azimuth=180
max_angle=55
altitude = 14.5
GCR = 0.28

""" Get weather data (real data from .csv file) 
Time, GHI, DHI, DNI, Solar zenith, Solar azimuth
From 2021-01-01 to 2021-12-31 approximately every 20sec (not regular)  """

real_weather = pd.read_csv('../../Risø Data.csv')#.iloc[0:2513]

# Set the timestamps as the new index

real_weather = real_weather.set_index('TmStamp')

real_weather.index.name = "utc_time"

real_weather.index = pd.to_datetime(real_weather.index)

# Resample to get a full day of data with 1 min resolution. The morning & evening values remain without irradiance data.

times = pd.date_range('2021-01-01', '2021-01-02', freq='1min', tz=tz)

real_weather = real_weather.reindex(times, method='nearest',limit=60)

GHI = real_weather["GHI"].fillna(0)

DHI = real_weather["DHI"].fillna(0)

DNI = real_weather["DNI"].fillna(0)

solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

apparent_zenith = solpos["apparent_zenith"]

azimuth = solpos["azimuth"]

""" For each time, calculate optimal angle of rotation (with 2° resolution) that yields the maximum POA
Use a transposition model to calculate POA irradiance from GHI, DNI and DHI """

beta_range = range(-max_angle, max_angle + 2, 2)

time_test = "2023-01-01 10:00:00+01:00"
   
POA_max = 0
beta_POA_max = 0
list_test = []

for beta in beta_range:

    POA_data = pvlib.irradiance.get_total_irradiance(beta, axis_azimuth, apparent_zenith[time_test], azimuth[time_test], DNI[time_test], GHI[time_test], DHI[time_test])
    POA_data = pvlib.irradiance.get_total_irradiance(beta, axis_azimuth, apparent_zenith[time_test], azimuth[time_test], DNI[time_test], GHI[time_test], DHI[time_test])
    POA_global = POA_data["poa_global"]
    
    list_test.append((beta,POA_global))
    
    if POA_global > POA_max:
        
        POA_max = POA_global
        beta_POA_max = beta



    
""" Comparison with true tracking """

truetracking_angles = pvlib.tracking.singleaxis(
apparent_zenith=apparent_zenith,
apparent_azimuth=azimuth,
axis_tilt=axis_tilt,
axis_azimuth=axis_azimuth,
max_angle=max_angle,
backtrack=False,  # for true-tracking
gcr=GCR)  # irrelevant for true-tracking

truetracking_position = truetracking_angles['tracker_theta'].fillna(0)

truetracking_position.plot(title='Truetracking Curve')


    
    