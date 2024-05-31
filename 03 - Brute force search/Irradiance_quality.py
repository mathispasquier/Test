# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:15:32 2024

@author: mpa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

"""

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import cos, sin, atan, radians, degrees, floor, sqrt, isnan

import pvanalytics

""" Input parameters"""

# Site parameters

tz='Etc/GMT-1'
latitude = 55.696166
longitude = 12.105216
altitude = 14.5
GCR = 0.28
albedo = 0.3

# Tracker & axis parameters

axis_tilt = 0 # Tracker axis tilt, in degrees
axis_azimuth = 180 # Tracker axis azimuth, in degrees
max_angle = 55 # Maximum angle for the tracker rotation, in degrees
U = 24 # Tracker voltage, in V
I = 2 # Tracker current, in A
w_min = 9 # Tracker rotation speed, in degrees/minute

# Data source: the file should contain 1 column with datetimes, 1 column with 'GHI' measurements and (if applicable) 1 column with 'DHI' measurements and 1 column with 'DNI' measurements

filename = r'C:\Users\mpa\OneDrive - EE\Documents\GitHub\2023\Risø Data formatted 2023.csv' # Name and path for the data file
filedate_beginning = '2023-01-01' # Beginning date in the data file
filedate_end = '2024-01-01' # End date in the data file
dataresolution = '1min' # Time step in the data file (or data frequency to use)

# Module parameters

module_params = {'alpha_sc': 0.0004,
                 'gamma_ref': 1.020,
                 'mu_gamma': -0.0002,
                 'I_L_ref': 17.27, #doubt -> Isc
                 'I_o_ref': 2.5e-10,
                 'R_sh_ref': 20000,
                 'R_sh_0': 80000, #doubt -> when running pvyst model, we get same electrical param. (Pmax in datasheet) 25C, 1000W
                 'R_s': 0.168,
                 'cells_in_series': 66,
                 'R_sh_exp': 5.5,
                 'EgRef': 1.12,
                 }

modules_in_series = 26 # Number of modules connected (in series) to the tracker
module_type = 'open_rack_glass_glass' # Module type
Pnom_module = 650 #  Module nominal power, in watts

# Simulation parameters

begin = '2023-01-01 00:00:00' # Beginning date for the simulation
end = '2024-01-01 00:00:00' # End date for the simulation
resolution = 15 # Tracker optimization resolution in minutes (if needed)
decomposition = False # Whether a decomposition model should be used or not
clearskymodel = False # Whether a clear sky model should be used or not
threshold_VI = 40 # Value for the variability index that corresponds to an hesitation factor of 1 in the TeraBase model

""" Getting irradiance data from a data source or calculating irradiance data using a model """

data_index = pd.date_range(filedate_beginning, filedate_end, freq=dataresolution, tz=tz)

# Creating a weather DataFrame that will contain irradiance data (GHI, DHI, DNI, DNI extraterrestrial, air mass)

if clearskymodel:
    weather = pd.DataFrame(data=None, index=data_index)
else:
    weather = pd.read_csv(filename)
    weather.index = data_index
    weather = weather.drop(['TmStamp'],axis=1)

# Select the days to rum the simulation

weather = weather.loc[begin:end]

""" Calculating sun & weather parameters for each period """

# Calculating the solar position for each period (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time)

solpos = pvlib.solarposition.get_solarposition(weather.index, latitude, longitude, altitude)

apparent_zenith = solpos['apparent_zenith']
zenith = solpos['zenith']
azimuth = solpos['azimuth']

# Calculating air mass and DNI extra used later in a transposition model

weather['DNI_extra'] = pvlib.irradiance.get_extra_radiation(weather.index)
weather['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith)

# Taking irradiance data from a decomposition model or a clearsky model, if applicable

linketurbidity = pvlib.clearsky.lookup_linke_turbidity(weather.index, latitude, longitude)
clearsky = pvlib.clearsky.ineichen(apparent_zenith, weather['airmass'], linketurbidity, altitude)

if decomposition:
    # Calculating DHI and DNI using GHI data and the Erbs decomposition model
    out_erbs = pvlib.irradiance.erbs(weather['GHI'], zenith, weather['GHI'].index)
    weather['DHI'] = out_erbs['dhi']
    weather['DNI'] = out_erbs['DNI']
elif clearskymodel:
    linketurbidity = pvlib.clearsky.lookup_linke_turbidity(weather.index, latitude, longitude)
    clearsky = pvlib.clearsky.ineichen(apparent_zenith, weather['airmass'], linketurbidity, altitude)
    weather['GHI'] = clearsky['ghi']
    weather['DHI'] = clearsky['dhi']
    weather['DNI'] = clearsky['dni'] 

test_limits = pvanalytics.quality.irradiance.check_irradiance_limits_qcrad(zenith, weather['DNI_extra'], weather['GHI'], weather['DHI'], weather['DNI'], limits=None)
consistency = pvanalytics.quality.irradiance.check_irradiance_consistency_qcrad(zenith, weather['GHI'], weather['DHI'], weather['DNI'], param=None)

data_quality = pd.DataFrame(data=None, index=data_index)
data_quality['zenith'] = zenith
data_quality['GHI_limits'] = test_limits[0]
data_quality['DHI_limits'] = test_limits[1]
data_quality['DNI_limits'] = test_limits[2]
data_quality['consistent_components'] = consistency[0]
data_quality['diffuse_ratio_limit'] = consistency[1]

data_quality.loc[data_quality[data_quality['zenith']>=90].index, 'consistent_components'] = True
data_quality.loc[data_quality[data_quality['zenith']>=90].index, 'diffuse_ratio_limit'] = True

issues = data_quality[(data_quality['GHI_limits']==False)|(data_quality['DHI_limits']==False)|(data_quality['DNI_limits']==False)|(data_quality['consistent_components']==False)|(data_quality['diffuse_ratio_limit']==False)].index

data_quality.to_csv(r'C:\Users\mpa\OneDrive - EE\Documents\GitHub\2023\Data quality 2023.csv',index=True,mode='w')

weather.loc[weather[weather['GHI']<0].index]