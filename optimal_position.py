# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

"""

import pvlib
import pandas as pd
from math import floor, sqrt
import numpy as np

""" Input parameters"""

# Site parameters

tz='Etc/GMT-1'
latitude = 55.696166
longitude = 12.105216
altitude = 14.5
GCR = 0.28
name = 'Risoe'

# Tracker & axis parameters

axis_tilt = 0 # Tracker axis tilt, in degrees
axis_azimuth = 180 # Tracker axis azimuth, in degrees
max_angle = 45 # Maximum angle for the tracker rotation, in degrees
w_min = 9 # Tracker rotation speed, in degrees/minute

# Optimization parameters

weather_data_path = 'C:/Users/mpa/OneDrive - EE/Documents/GitHub/Risø implementation/Data/weather.csv' # Path to the local weather file that contains irradiance data from the FTP
resolution = 15 # Tracker optimization resolution in minutes (if needed)
threshold_VI = 40 # Value for the variability index that corresponds to an hesitation factor of 1 in the TeraBase model

""" Getting irradiance data from a data source """

weather = pd.read_csv(weather_data_path)
weather = weather.iloc[-resolution:] # Get weather data for the last 15 minutes only
weather.rename(columns={'Global_Horizontal_Pyr_Avg': 'GHI', 'Diffuse_Horizontal_Avg': 'DHI', 'Direct_Normal_Avg': 'DNI'}, inplace=True)
weather.set_index('TIMESTAMP', inplace=True)

# Change the index to make it timezone aware
weather.index = pd.to_datetime(weather.index)
weather.index = weather.index.tz_localize(tz)

""" Calculating sun & weather parameters for each period """

# Calculating the solar position at the current time (weather.index is timezone aware)

solpos = pvlib.solarposition.get_solarposition(weather.index, latitude, longitude, altitude)

# Calculating air mass and DNI extra used later in a transposition model

weather['DNI_extra'] = pvlib.irradiance.get_extra_radiation(weather.index)
weather['airmass'] = pvlib.atmosphere.get_relative_airmass(solpos['zenith'])

# Calculating variability

linketurbidity = pvlib.clearsky.lookup_linke_turbidity(weather.index, latitude, longitude)
clearsky = pvlib.clearsky.ineichen(solpos['apparent_zenith'], weather['airmass'], linketurbidity, altitude)
     
num = 0
den = 0
        
for k in range(1,resolution):
            
    num += (weather['GHI'].iloc[k]-weather['GHI'].iloc[k-1])**2 + 1**2 
    den += (clearsky['ghi'].iloc[k]-clearsky['ghi'].iloc[k-1])**2 + 1**2 
        
variability = sqrt(num)/sqrt(den) # variability of the last 15 minutes

""" FUNCTIONS """

def find_max_angle_backtracking(astronomical_tracking_angle_backtracking, astronomical_tracking_angle_no_backtracking, max_angle):
    """
    
    Returns the limit backtracking angle (absolute value): minimum between the backtracking angle (absolute value) and the tracker maximum angle of rotation.
    The input files should have the same time index.

    Parameters
    ----------
    astronomical_tracking_angles_backtracking : float
        Astronomical angle calculated (backtracking is implemented).
    astronomical_tracking_angles_no_backtracking : float
        Astronomical angle calculated (backtracking is not implemented).
    max_angle : int
        Tracker maximum rotation angle.

    Returns
    -------
    max_angle_backtracking : float
        Floor value (absolute value) of the limit angle for the tracker rotation angle.

    """
    
    if astronomical_tracking_angle_backtracking != astronomical_tracking_angle_no_backtracking:
        max_angle_backtracking = floor(abs(astronomical_tracking_angle_backtracking))
    else:
        max_angle_backtracking = max_angle
        
    return max_angle_backtracking

def model_brute_force_search(GHI, DHI, DNI, DNI_extra, airmass, current_angle, apparent_zenith, azimuth, max_angle_backtracking, w_min, resolution=1):
    """
    Calculates the optimized tracker rotation angle using the brute force search model. 
    For all the possible angles of rotation, a transposition model (e.g. Perez model) calculates the associated POA irradiance. 
    The angle that maximizes the POA irradiance is the optimized tracker rotation angle.
    From its current position, the tracker can only move by +/- its rotation speed.
    
    Parameters
    ----------
    GHI: float
        Current GHI
    DHI: float
        Current DHI
    DNI: float
        Current DNI
    DNI_extra: float
        Current DNI_extra
    airmass: float
        Current air mass
    apparent_zenith : float
        Current apparent zenith
    azimuth : float
        Current azimuth
    max_angle_backtracking : float
        Floor value (absolute value) of the limit angle for the tracker rotation angle.
    w_min : int or float
        Tracker rotation speed, in minutes.
    resolution : Tracker optimization resolution in minutes, optional
        Timestep to calculate the optimal angle and to move the tracker. The default is 1.

    Returns
    -------
    brute_force_search : DataFrame(index: DatetimeIndex, angle: int)
        Optimized tracker rotation angle calculated using the brute force search method.

    """
 
    max_irradiance = 0
    optimal_angle = 0
        
    min_angle_range = max(-max_angle_backtracking,current_angle - w_min*resolution)
    max_angle_range = min(max_angle_backtracking,current_angle + w_min*resolution)
        
    for angle in range(min_angle_range,max_angle_range,1):
            
        if angle < 0:
            surface_azimuth = 90 
            angle_abs = abs(angle)
        else:
            surface_azimuth = 270
            angle_abs = angle
            

        total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                            surface_azimuth=surface_azimuth, 
                                                            dni=DNI, 
                                                            ghi=GHI, 
                                                            dhi=DHI, 
                                                            solar_zenith=apparent_zenith, 
                                                            solar_azimuth=azimuth,
                                                            dni_extra=DNI_extra,
                                                            airmass=airmass,
                                                            model='perez',
                                                            model_perez='allsitescomposite1990') 
        total_irradiance = total_irrad['poa_global']
                       
            
        if total_irradiance > max_irradiance:
            max_irradiance = total_irradiance
            optimal_angle = angle
    
    return optimal_angle

def TeraBase(astronomical_tracking_position, tracking_position, variability, w_min, threshold_VI, resolution=1):
    """
    The TeraBase model calculates a corrected tracker rotation angle between the astronomical angle and the optimal tracker rotation angle (calculated with a model).
    The calculations takes into account a movement penalty, which represents the amount of time a tracker must spend in transit from the previous rotation angle to the current rotation angle and is a percentage of the total amount of time encompassed by that timestamp.
    The hesitation factor is an empirical term which represents controls hesitancy to go to the optimal tracker rotation angle.

    Parameters
    ----------
    astronomical_tracking_position : float
        Astronomical angle calculated (backtracking is implemented).
    tracking_position : 
        Optimal tracker rotation angle calculated using the brute force search model.
    variability : float
        Variability of the last 15 minutes used to calculate the hesitation factor.
    w_min : int or float
        Tracker rotation speed, in minutes.
    threshold_VI : int
        Variability index that corresponds to an heistation factor of 1
    resolution : int, optional
        Tracker optimization resolution in minutes. The default is 1.

    Returns
    -------
    TeraBase : float
        Tracker rotation angle calculated using an optimal angle model and then corrected with the TeraBase model.

    """

    hesitation_factor = variability/threshold_VI
        
    movement_penalty = abs(astronomical_tracking_position-tracking_position)/(w_min*resolution) # w_min is the tracker rotation speed (in min.) and resolution is the number of minutes in the timestep
    movement_penalty = min(movement_penalty, 1)
        
    # The maximum hesitation factor is 1-movement_penalty
    hesitation_factor_parameter = min(hesitation_factor,1-movement_penalty)
        
    corrected_angle = ((1-movement_penalty/100-hesitation_factor_parameter)*tracking_position)+((movement_penalty/100)*0.5*(astronomical_tracking_position+tracking_position))+(hesitation_factor_parameter*astronomical_tracking_position)
    
    return corrected_angle

""" Calculation of maximal angle (taking into account the backtracking limit) """

# Calculate astronomical tracking angles with and without backtracking

current_apparent_zenith = solpos['apparent_zenith'].iloc[-1]
current_azimuth = solpos['azimuth'].iloc[-1]
astronomical_tracking_angle_no_backtracking = pvlib.tracking.singleaxis(
    apparent_zenith=current_apparent_zenith,
    apparent_azimuth=current_azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False, 
    gcr=GCR)
astronomical_tracking_angle_no_backtracking = astronomical_tracking_angle_no_backtracking['tracker_theta'][0]
if np.isnan(astronomical_tracking_angle_no_backtracking): astronomical_tracking_angle_no_backtracking = 0.0
    
astronomical_tracking_angle = pvlib.tracking.singleaxis(
    apparent_zenith=current_apparent_zenith,
    apparent_azimuth=current_azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=True, 
    gcr=GCR)
astronomical_tracking_angle = astronomical_tracking_angle['tracker_theta'][0]
if np.isnan(astronomical_tracking_angle): astronomical_tracking_angle = 0.0

# Calculating the limit backtracking angle

max_angle_backtracking = find_max_angle_backtracking(astronomical_tracking_angle, astronomical_tracking_angle_no_backtracking, max_angle)

# Calculating the best angle from the brute force search model

current_angle = astronomical_tracking_angle # CHANGE: take the current angle from the TCU

GHI = weather['GHI'].iloc[-1]
DHI = weather['DHI'].iloc[-1]
DNI = weather['DNI'].iloc[-1]
DNI_extra = weather['DNI_extra'].iloc[-1]
airmass = weather['airmass'].iloc[-1]

optimal_angle = model_brute_force_search(GHI, DHI, DNI, DNI_extra, airmass, current_angle, current_apparent_zenith, current_azimuth, max_angle_backtracking, w_min, resolution)

# Calculating the target angle using the TeraBase model

target_angle = TeraBase(astronomical_tracking_angle, optimal_angle, variability, w_min, threshold_VI, resolution)