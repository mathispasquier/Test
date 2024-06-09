# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

Comparison of backtracking curves for different GCR & maximum angles of rotation
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
name = 'Risø'
axis_tilt=0
axis_azimuth=180
max_angle=55
altitude = 14.5
GCR = 0.28

font_properties = {
    'font.family': 'Segoe UI',
}

# Update rcParams
plt.rcParams.update(font_properties)
plt.rcParams.update({
    'axes.labelsize': 11,            # Axis labels size
    'axes.labelweight': 'normal',      # Axis labels weight (bold)
    'axes.titlesize': 13,            # Title size
    'legend.fontsize': 9,           # Legend size
    'font.size': 9,                 # Default font size for text
    'figure.autolayout': True,
    'lines.linewidth': 1,
})

""" Calculation of the rotation angle - Library for modeling tilt angles for single-axis tracker arrays """

times = pd.date_range('2023-06-01', '2023-06-02', freq='1min', tz=tz)

solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

cm = 1/2.54
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(12*cm, 9*cm))

# True-tracking

truetracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=solpos['apparent_zenith'],
    apparent_azimuth=solpos['azimuth'],
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False,
    gcr=GCR)

truetracking_position = truetracking_angles['tracker_theta'].fillna(0)
truetracking_position.plot(label='No backtracking', ax=ax, color='navy')

# Backtracking

for gcr in [0.28, 0.4, 0.6]:
    

    backtracking_angles = pvlib.tracking.singleaxis(
        apparent_zenith=solpos['apparent_zenith'],
        apparent_azimuth=solpos['azimuth'],
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=True,
        gcr=gcr)

    backtracking_position = backtracking_angles['tracker_theta'].fillna(0)
    if gcr==0.28:
        backtracking_position.plot(label=f'GCR:{gcr}',
                               ax=ax,
                               color='green')
    elif gcr==0.4:
        backtracking_position.plot(label=f'GCR:{gcr}',
                               ax=ax,color='darkred')    
    else:
        backtracking_position.plot(label=f'GCR:{gcr}',
                               ax=ax,color='darkorange')
        
plt.ylabel('Tracker Tilt (°)')
plt.grid()
plt.legend()
plt.show() 
    