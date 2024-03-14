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

tz='Etc/GMT-1'
latitude = 55.696166
longitude = 12.105216
name = 'Risoe'
axis_tilt=0
axis_azimuth=180
max_angle=55
altitude = 14.5
GCR = 0.28
U = 24 # in V
I = 2 # in A
w = 9/60 # in degree/second

""" Get weather data (real data from .csv file) """

real_weather = pd.read_csv('../../Risø Data formatted_2.csv')

real_weather.index = pd.date_range('2023-01-01', '2024-01-01', freq='1min', tz=tz)
real_weather = real_weather.drop(['TmStamp'],axis=1)

""" Select the days when to calculate via brute force search """

begin = '2023-01-01 00:00:00'
end = '2024-01-01 00:00:00'

real_weather = real_weather.loc[begin:end]

""" Calculate the solar position for each time """

solpos = pvlib.solarposition.get_solarposition(real_weather.index, latitude, longitude, altitude)

apparent_zenith = solpos["apparent_zenith"]
zenith = solpos["zenith"]
azimuth = solpos["azimuth"]

""" Calculate air mass and DNI extra for the Perez transposition model """

real_weather["DNI_extra"] = pvlib.irradiance.get_extra_radiation(real_weather.index)
real_weather["airmass"] = pvlib.atmosphere.get_relative_airmass(zenith)

DNI_extra = real_weather["DNI_extra"]
airmass = real_weather["airmass"]

linketurbidity = pvlib.clearsky.lookup_linke_turbidity(real_weather.index, latitude, longitude)

GHI = real_weather["GHI"] 
DHI = real_weather["DHI"] 
DNI = real_weather["DNI"] 

""" Decomposition model
The Erbs method, erbs() developed by Daryl Gregory Erbs at the University of Wisconsin in 1982 is a piecewise 
correlation that splits kt into 3 regions: linear for kt <= 0.22, a 4th order polynomial between 0.22 < kt <= 0.8, 
and a horizontal line for kt > 0.8. """

out_erbs = pvlib.irradiance.erbs(GHI, zenith, GHI.index)

DHI_erbs = out_erbs["dhi"]
DNI_erbs = out_erbs["dni"]

real_weather["DHI_erbs"] = DHI_erbs
real_weather["DNI_erbs"] = DNI_erbs

""" For each time, calculate optimal angle of rotation (with 2° resolution) that yields the maximum POA
Use a transposition model to calculate POA irradiance from GHI, DNI and DHI """

def find_optimal_rotation_angle(ghi, dhi, dni, dni_extra, airmass, solar_position):
    """
    Find the optimal rotation angle within given limits.
    """
    diffuse_tracking = pd.DataFrame(data=None, index=GHI.index)
    diffuse_tracking['angle'] = 0.0
    diffuse_tracking['POA global'] = 0.0
    diffuse_tracking['degrees_moved'] = 0.0
    optimal_angle = 0
    previous_angle = 0
    
    for i in range(ghi.index.size):
        
        max_irradiance = 0

        for angle in range(-55,56,2):
            
            if angle < 0:
                surface_azimuth = 90 
                angle_abs = abs(angle)
            else:
                surface_azimuth = 270
                angle_abs = angle
            
            total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                                 surface_azimuth=surface_azimuth, 
                                                                 dni=dni.iloc[i], 
                                                                 ghi=ghi.iloc[i], 
                                                                 dhi=dhi.iloc[i], 
                                                                 solar_zenith=solar_position['apparent_zenith'].iloc[i], 
                                                                 solar_azimuth=solar_position['azimuth'].iloc[i],
                                                                 dni_extra=dni_extra.iloc[i],
                                                                 airmass=airmass.iloc[i],
                                                                 model='perez',
                                                                 model_perez='allsitescomposite1990')
            total_irradiance = total_irrad['poa_global']
            
            if total_irradiance > max_irradiance:
                max_irradiance = total_irradiance
                optimal_angle = angle
        
        if max_irradiance == 0:
            optimal_angle = 0
        
        diffuse_tracking['angle'].iloc[i] = optimal_angle
        diffuse_tracking['degrees_moved'].iloc[i] = abs(optimal_angle-previous_angle)
        previous_angle = optimal_angle
        diffuse_tracking['POA global'].iloc[i] = max_irradiance
    
    return diffuse_tracking

diffuse_tracking = find_optimal_rotation_angle(GHI, DHI, DNI, DNI_extra, airmass, solpos)
diffuse_tracking_erbs = find_optimal_rotation_angle(GHI, DHI_erbs, DNI_erbs, DNI_extra, airmass, solpos)

# KPIs diffuse tracking

total_degrees_diffuse = diffuse_tracking['degrees_moved'].sum()
delta_t_diffuse = total_degrees_diffuse/w # Total time (in s) during which the tracker moved
consumption_tracker_diffuse = U*I*delta_t_diffuse/3600000 # in kWh
energy_yield_diffuse = (diffuse_tracking['POA global'].mean()/1000)*(diffuse_tracking.index.size/60) # Average POA irradiance in kW * number of hours

total_degrees_diffuse_erbs = diffuse_tracking_erbs['degrees_moved'].sum()
delta_t_diffuse_erbs = total_degrees_diffuse_erbs/w # Total time (in s) during which the tracker moved
consumption_tracker_diffuse_erbs = U*I*delta_t_diffuse_erbs/3600000 # in kWh
energy_yield_diffuse_erbs = (diffuse_tracking_erbs['POA global'].mean()/1000)*(diffuse_tracking_erbs.index.size/60) # Average POA irradiance in kW * number of hours

""" Comparison with true tracking """

truetracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=apparent_zenith,
    apparent_azimuth=azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False,  # for true-tracking
    gcr=GCR)  # irrelevant for true-tracking

# Calculation of POA irradiance for true tracking

true_tracking = pd.DataFrame(data=None, index=GHI.index)
true_tracking['tracker_theta'] = 0.0
true_tracking['tracker_theta'] = truetracking_angles['tracker_theta'].fillna(0)
true_tracking['POA global'] = 0.0
true_tracking['degrees_moved'] = 0.0

previous_angle = 0

for i in range(true_tracking.index.size):
    
    angle = true_tracking["tracker_theta"].iloc[i]
        
    if angle < 0:
        surface_azimuth = 90 
        angle_abs = abs(angle)
    else:
        surface_azimuth = 270
        angle_abs = angle

    total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                        surface_azimuth=surface_azimuth, 
                                                        dni=real_weather["DNI"].iloc[i], 
                                                        ghi=real_weather["GHI"].iloc[i], 
                                                        dhi=real_weather["DHI"].iloc[i], 
                                                        solar_zenith=solpos['apparent_zenith'].iloc[i], 
                                                        solar_azimuth=solpos['azimuth'].iloc[i],
                                                        dni_extra=real_weather["DNI_extra"].iloc[i],
                                                        airmass=real_weather["airmass"].iloc[i],
                                                        model='perez',
                                                        model_perez='allsitescomposite1990')
    
    true_tracking["POA global"].iloc[i] = total_irrad["poa_global"]
    true_tracking['degrees_moved'].iloc[i] = abs(previous_angle - angle)
    previous_angle = angle

# KPIs astronomical tracking

total_degrees_astronomical = true_tracking['degrees_moved'].sum()
delta_t_astronomical = total_degrees_astronomical/w # Total time (in s) during which the tracker moved
consumption_tracker_astronomical = U*I*delta_t_astronomical/3600000 # in kWh
energy_yield_astronomical = (true_tracking['POA global'].mean()/1000)*(true_tracking.index.size/60) # Average POA irradiance in kW * number of hours

""" Plot data """

# Irradiance GHI, DNI, DHI

"""GHI.plot(title='Actual irradiance data', label="GHI")
DNI.plot(title='Actual irradiance data', label="DNI")
#DHI.plot(title='Actual irradiance data', label="DHI")
DNI_erbs.plot(title='Actual irradiance data', label="DNI Erbs", linestyle='--')
#DHI_erbs.plot(title='Actual irradiance data', label="DHI Erbs")

# Tracking curves & POA irradiance

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

diffuse_tracking['angle'].plot(title='Tracking Curve', label="Optimal tracking", ax=axes[0])
true_tracking["tracker_theta"].plot(title='Tracking Curve', label="Astronomical tracking",ax=axes[0])

diffuse_tracking['POA global'].plot(title='Irradiance', label="POA diffuse tracking", ax=axes[1])
true_tracking["POA global"].plot(title='Irradiance', label="POA astronomical tracking", ax=axes[1])

axes[0].legend(title="Tracker Tilt")
axes[1].legend(title="Irradiance")


plt.legend()
plt.show()"""
    
    