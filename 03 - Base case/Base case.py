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

real_weather = pd.read_csv('../../Risø Data formatted.csv')

# Set the timestamps as the new index

real_weather = real_weather.set_index('TmStamp')

real_weather.index.name = "utc_time"

real_weather.index = pd.to_datetime(real_weather.index)

# Select the days

m = 160 # Day when it begins
n = 165 # Day when it ends

real_weather = real_weather.iloc[(m-1)*1440:(n-1)*1440]

GHI = real_weather["GHI"]
DHI = real_weather["DHI"]
DNI = real_weather["DNI"]

""" Calculate the solar position for each time """

solpos = pvlib.solarposition.get_solarposition(real_weather.index, latitude, longitude, altitude)

apparent_zenith = solpos["apparent_zenith"]

azimuth = solpos["azimuth"]

""" For each time, calculate optimal angle of rotation (with 2° resolution) that yields the maximum POA
Use a transposition model to calculate POA irradiance from GHI, DNI and DHI """

brute_force_search = pd.DataFrame(data=None, index=real_weather.index)
brute_force_search["beta_opt"] = 0
brute_force_search["POA_global_opt"] = 0

beta_range = range(-max_angle, max_angle + 2, 2)

for time, data in real_weather.iterrows():
    
    POA_max = 0
    beta_POA_max = 0
    
    for beta in beta_range:
    
        # Transposition model 
        
        POA_data = pvlib.irradiance.get_total_irradiance(beta, axis_azimuth, apparent_zenith[time], azimuth[time], DNI[time], GHI[time], DHI[time])
        POA_global = POA_data["poa_global"]
        
        # Definition of the new optimal angle when the associated POA is maximal
        
        if POA_global > POA_max:
            
            POA_max = POA_global
            beta_POA_max = beta
    
    brute_force_search.loc[time,"beta_opt"] = beta_POA_max
    brute_force_search.loc[time,"POA_global_opt"] = POA_max
        
""" Comparison with true tracking """

truetracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=apparent_zenith,
    apparent_azimuth=azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False,  # for true-tracking
    gcr=GCR)  # irrelevant for true-tracking

fig, ax = plt.subplots()

truetracking_position = truetracking_angles['tracker_theta'].fillna(0)

optimal_angle = brute_force_search["beta_opt"]
optimal_angle.index = truetracking_position.index

truetracking_position.plot(title='Tracking Curve', label="Astronomical tracking",ax=ax)
optimal_angle.plot(title='Tracking Curve', label="Optimal tracking", ax=ax)

plt.legend()
plt.show() 
    
    