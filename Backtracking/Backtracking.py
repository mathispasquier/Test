# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa
"""

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

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

""" Get weather data (typical year) """

weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude)[0]
weather.index.name = "utc_time"
timeseries = weather.index

""" Calculation of the rotation angle - Library for modeling tilt angles for single-axis tracker arrays """

times = pd.date_range('2019-01-01', '2019-01-02', freq='5min', tz=tz)

solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

fig, ax = plt.subplots()

truetracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=solpos['apparent_zenith'],
    apparent_azimuth=solpos['azimuth'],
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False,
    gcr=GCR)

truetracking_position = truetracking_angles['tracker_theta'].fillna(0)

#truetracking_position.plot(title='Truetracking Curve',
                           #ax=ax)

for gcr in [0.2, 0.4, 0.6]:
    backtracking_angles = pvlib.tracking.singleaxis(
        apparent_zenith=solpos['apparent_zenith'],
        apparent_azimuth=solpos['azimuth'],
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=True,
        gcr=gcr)

    backtracking_position = backtracking_angles['tracker_theta'].fillna(0)
    backtracking_position.plot(title='Backtracking Curve - maximum angle 55°',
                               label=f'GCR:{gcr:0.01f}',
                               ax=ax)

plt.legend()
plt.show()

""" Estimation of the energy produced via ModelChain """


location = pvlib.location.Location(
    latitude, 
    longitude, 
    tz, 
    altitude, 
    name)

mount = pvlib.pvsystem.SingleAxisTrackerMount(
    axis_tilt,
    axis_azimuth,
    max_angle,
    False)

test = mount.get_orientation(solpos['apparent_zenith'], 
                             solpos['azimuth'])

array = pvlib.pvsystem.Array(mount=mount, 
                             module_parameters=module, 
                             temperature_model_parameters=temperature_model_parameters)

system = pvlib.pvsystem.PVSystem(arrays=[array], 
                                 inverter_parameters=inverter)

mc = pvlib.modelchain.ModelChain(system, location)

mc.run_model(weather)

annual_energy = mc.results.ac.sum()
    
    