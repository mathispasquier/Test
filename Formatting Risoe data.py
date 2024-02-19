# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:14:44 2024

@author: mathi
"""

import csv

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import cos, sin, atan, atan2, radians, degrees

tz = 'Europe/Copenhagen'

real_weather = pd.read_csv('../Risø Data.csv')

# Set the timestamps as the new index

real_weather = real_weather.set_index('TmStamp')

real_weather.index.name = "utc_time"

real_weather.index = pd.to_datetime(real_weather.index)

# Fill the gaps in the data with the nearest value

real_weather = real_weather.reindex(real_weather.index, method='nearest')

# Resample to get a full day of data with 1 min resolution. The morning & evening values remain without irradiance data.

times = pd.date_range('2021-01-01', '2022-01-01', freq='1min', tz=tz)

real_weather = real_weather.reindex(times, method=None)

real_weather["GHI"] = real_weather["GHI"].fillna(0)
real_weather["DHI"] = real_weather["DHI"].fillna(0)
real_weather["DNI"] = real_weather["DNI"].fillna(0)

real_weather = real_weather[["GHI","DHI","DNI"]]
real_weather.to_csv('../Risø Data formatted.csv', index=True)