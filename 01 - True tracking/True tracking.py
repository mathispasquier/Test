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

#times = pd.date_range('2019-01-01', '2019-01-02', freq='5min', tz=tz)

solpos = pvlib.solarposition.get_solarposition(timeseries, latitude, longitude, altitude)

truetracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=solpos['apparent_zenith'],
    apparent_azimuth=solpos['azimuth'],
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False,  # for true-tracking
    gcr=0.5)  # irrelevant for true-tracking

truetracking_position = truetracking_angles['tracker_theta'].fillna(0)


truetracking_position_1d = truetracking_position[0:23]
truetracking_position_1d.plot(title='Truetracking Curve')

plt.show()

""" Calculation of the rotation angle - using equations from Anderson and Mikofski"""

solar_zenith = solpos['apparent_zenith'].apply(radians).to_numpy()
solar_elevation = solpos['apparent_elevation'].apply(radians).to_numpy()
solar_azimuth = solpos['azimuth'].apply(radians).to_numpy()

Rx = np.array([[1,0,0],
              [0,cos(radians(axis_tilt)),-sin(radians(axis_tilt))],
              [0,sin(radians(axis_tilt)),cos(radians(axis_tilt))]])

Rz = np.array([[cos(radians(axis_azimuth)),-sin(radians(axis_azimuth)),0],
              [sin(radians(axis_azimuth)),cos(radians(axis_azimuth)),0],
              [0,0,1]])

rotation_angles_anderson = []


for i in range(0,8760):
    s = [cos(solar_elevation[i])*sin(solar_azimuth[i]),cos(solar_elevation[i])*cos(solar_azimuth[i]),sin(solar_elevation[i])]
    s_prime = Rx@Rz@s
    theta = degrees(atan2(s_prime[0],s_prime[2]))
    if theta < -max_angle or theta > max_angle:
        theta = 0
    rotation_angles_anderson.append(theta)


""" Calculation of the rotation angle - using equations from Marion and Dobos"""

rotation_angles_marion = []

for i in range(0,8760):
    a = (sin(solar_zenith[i])*sin(solar_azimuth[i]-radians(axis_azimuth)))
    b = (sin(solar_zenith[i])*cos(solar_azimuth[i]-radians(axis_azimuth))*sin(radians(axis_tilt))+cos(solar_zenith[i])*cos(radians(axis_tilt)))
    x = a/b
    theta = degrees(atan2(a,b))
    if theta < -max_angle or theta > max_angle:
        theta = 0
    rotation_angles_marion.append(theta)
    