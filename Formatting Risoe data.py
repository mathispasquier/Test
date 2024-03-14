# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:14:44 2024

@author: mathi
"""

import datetime

import csv

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import cos, sin, atan, atan2, radians, degrees

tz = 'Europe/Copenhagen'

index_data = pd.read_csv('C:/Users/mpa/OneDrive - EE/Documents/GitHub/index2023.csv')

real_weather = pd.read_csv('C:/Users/mpa/OneDrive - EE/Documents/GitHub/post_qc_2023.csv')

# Resample to fill the missing values during the day (communication issues) with the nearest value

index_data = index_data.set_index('TmStamp')

index_data.index.name = "utc_time"

index_data.index = pd.to_datetime(index_data.index)

real_weather = real_weather.set_index('TmStamp')

real_weather.index.name = "utc_time"

real_weather.index = pd.to_datetime(real_weather.index)


real_weather = real_weather.reindex(index_data.index, method='nearest')

# Resample to get a full day of data with 1 min resolution. The morning & evening values remain without irradiance data.

times = pd.date_range('2023-01-01', '2024-01-01', freq='1min', tz=tz)

real_weather = real_weather.reindex(times, method=None)

real_weather["GHI"] = real_weather["GHI"].fillna(0)
real_weather["DHI"] = real_weather["DHI"].fillna(0)
real_weather["DNI"] = real_weather["DNI"].fillna(0)

real_weather.to_csv('../Risø Data formatted_2.csv', index=True)