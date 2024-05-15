# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

"""

import pvlib
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from math import cos, sin, atan, radians, degrees, floor, sqrt
import pytz

""" Input parameters"""

# Site parameters

tz='Etc/GMT-1'
latitude = 55.696166
longitude = 12.105216
altitude = 14.5
GCR = 0.28
albedo = 0.3
name = 'Risoe'

# Tracker & axis parameters

axis_tilt = 0 # Tracker axis tilt, in degrees
axis_azimuth = 180 # Tracker axis azimuth, in degrees
max_angle = 45 # Maximum angle for the tracker rotation, in degrees
U = 24 # Tracker voltage, in V
I = 2 # Tracker current, in A
w_min = 9 # Tracker rotation speed, in degrees/minute
eta = 0.4 # Hesitation factor for the TeraBase model

# Optimization parameters

resolution = 15 # Tracker optimization resolution in minutes (if needed)
threshold_VI = 40 # Value for the variability index that corresponds to an hesitation factor of 1 in the TeraBase model

""" Getting irradiance data from a data source """

calculateAngle = True

# while calculateAngle:
    
#     currentTime = datetime.datetime.utcnow()
    
#     # Update every 15 min
#     if currentTime.minute%15 == 0:
        
#         # Path to data file
#         weather = pd.read_csv('C:/Users/mpa/OneDrive - EE/Documents/GitHub/Risø implementation/Data/weather.csv')
#         weather = weather[-15:]
#         weather.rename(columns={'Global_Horizontal_Pyr_Avg': 'GHI', 'Diffuse_Horizontal_Avg': 'DHI', 'Direct_Normal_Avg': 'DNI'}, inplace=True)


    
currentTime = datetime.datetime.utcnow()


# Get weather data
weather = pd.read_csv('C:/Users/mpa/OneDrive - EE/Documents/GitHub/Risø implementation/Data/weather.csv')
weather = weather.iloc[-15:]
weather.rename(columns={'Global_Horizontal_Pyr_Avg': 'GHI', 'Diffuse_Horizontal_Avg': 'DHI', 'Direct_Normal_Avg': 'DNI'}, inplace=True)
weather.set_index('TIMESTAMP', inplace=True)
weather.index = pd.to_datetime(weather.index)
weather.index = weather.index.tz_localize(tz)

""" Calculating sun & weather parameters for each period """

# Calculating the solar position for each period (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time)

solpos = pvlib.solarposition.get_solarposition(weather.index, latitude, longitude, altitude)

apparent_zenith = solpos['apparent_zenith']
zenith = solpos['zenith']
azimuth = solpos['azimuth']

# Calculating air mass and DNI extra used later in a transposition model

weather['DNI_extra'] = pvlib.irradiance.get_extra_radiation(weather.index)
weather['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith)

# Calculating variability

linketurbidity = pvlib.clearsky.lookup_linke_turbidity(weather.index, latitude, longitude)
clearsky = pvlib.clearsky.ineichen(apparent_zenith, weather['airmass'], linketurbidity, altitude)

weather['Variability GHI'] = 1.0
     
num = 0
den = 0
        
for k in range(1,15):
            
    num += (weather['GHI'].iloc[k]-weather['GHI'].iloc[k-1])**2 + 1**2 
    den += (clearsky['ghi'].iloc[k]-clearsky['ghi'].iloc[k-1])**2 + 1**2 
        
    weather.loc[weather.index[14], 'Variability GHI'] = sqrt(num)/sqrt(den)

""" IMPLEMENTATION OF FUNCTIONS FOR DIFFERENT MODELS """

""" Calculation of maximum angle due to backtracking """

def find_max_angle_backtracking(astronomical_tracking_angles_backtracking, astronomical_tracking_angles_no_backtracking, max_angle):
    """
    
    Returns the limit backtracking angle (absolute value): minimum between the backtracking angle (absolute value) and the tracker maximum angle of rotation.
    The input files should have the same time index.

    Parameters
    ----------
    astronomical_tracking_angles_backtracking : DataFrame(index: DatetimeIndex, tracker_theta: float)
        Astronomical angle calculated (backtracking is implemented).
    astronomical_tracking_angles_no_backtracking : DataFrame(index: DatetimeIndex, tracker_theta: float)
        Astronomical angle calculated (backtracking is not implemented).
    max_angle : int
        Tracker maximum rotation angle.

    Returns
    -------
    max_angle_backtracking : DataFrame(index: DatetimeIndex, angle: int)
        Floor value (absolute value) of the limit angle for the tracker rotation angle.

    """

    max_angle_backtracking = pd.DataFrame(data=None, index=astronomical_tracking_angles_backtracking.index)
    max_angle_backtracking['angle'] = 0
    
    for i in range(astronomical_tracking_angles_backtracking.index.size):
        
        if astronomical_tracking_angles_backtracking['tracker_theta'].iloc[i] != astronomical_tracking_angles_no_backtracking['tracker_theta'].iloc[i]:
            max_angle_backtracking['angle'].iloc[i] = floor(abs(astronomical_tracking_angles_backtracking['tracker_theta'].iloc[i]))
        else:
            max_angle_backtracking['angle'].iloc[i] = max_angle
        
    return max_angle_backtracking

""" Optimal angle models: brute force search, CENER, and binary mode """

def model_brute_force_search(weather, previous_angle, solar_position, max_angle_backtracking, w_min, resolution=1):
    """
    Calculates the optimized tracker rotation angle using the brute force search model. 
    For all the possible angles of rotation, a transposition model (e.g. Perez model) calculates the associated POA irradiance. 
    The angle that maximizes the POA irradiance is the optimized tracker rotation angle.
    From its previous position, the tracker can only move by +/- its rotation speed.
    
    Parameters
    ----------
    weather : DataFrame(index: DatetimeIndex, GHI: float, DHI: float, DNI: float, DNI_extra: float, airmass: float)
        Weather DataFrame that contains irradiance data (GHI, DHI, DNI, DNI extraterrestrial) and air mass.
    solar_position : DataFrame(index: DatetimeIndex, apparent_zenith: float, zenith: float, apparent_elevation: float, elevation: float, azimuth: float, equation_of_time: float)
        DataFrame that contains the parameters of the solar position (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time).
    max_angle_backtracking : DataFrame(index: DatetimeIndex, angle: int
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
        
    min_angle_range = max(-max_angle_backtracking,previous_angle - w_min*resolution)
    max_angle_range = min(max_angle_backtracking,previous_angle + w_min*resolution)
        
    for angle in range(min_angle_range,max_angle_range,1):
            
        if angle < 0:
            surface_azimuth = 90 
            angle_abs = abs(angle)
        else:
            surface_azimuth = 270
            angle_abs = angle
            

        total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                            surface_azimuth=surface_azimuth, 
                                                            dni=weather['DNI'], 
                                                            ghi=weather['GHI'], 
                                                            dhi=weather['DHI'], 
                                                            solar_zenith=solar_position['apparent_zenith'], 
                                                            solar_azimuth=solar_position['azimuth'],
                                                            dni_extra=weather['DNI_extra'],
                                                            airmass=weather['airmass'],
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
    astronomical_tracking : DataFrame(index: DatetimeIndex, angle: int)
        Astronomical angle calculated (backtracking is implemented). The angle is the floor value of the theoretical astronomical angle.
    tracking : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model.
    hesitation_factor : float
        Hesitation factor used in the TeraBase model.
    w_min : int or float
        Tracker rotation speed, in minutes.
    threshold_VI : int
        Variability index that corresponds to an heistation factor of 1
    resolution : int, optional
        Tracker optimization resolution in minutes. The default is 1.

    Returns
    -------
    TeraBase : DataFrame(index: DatetimeIndex, angle: int)
        Tracker rotation angle calculated using an optimal angle model and then corrected with the TeraBase model.

    """

    hesitation_factor = variability/threshold_VI
        
    movement_penalty = abs(astronomical_tracking_position-tracking_position)/(w_min*resolution) # w_min is the tracker rotation speed (in min.) and resolution is the number of minutes in the timestep
    movement_penalty = min(movement_penalty, 1)
        
    # The maximum hesitation factor is 1-movement_penalty
    hesitation_factor_parameter = min(hesitation_factor,1-movement_penalty)
        
    corrected_angle = ((1-movement_penalty/100-hesitation_factor_parameter)*tracking_position)+((movement_penalty/100)*0.5*(astronomical_tracking_position+tracking_position))+(hesitation_factor_parameter*astronomical_tracking_position)
    
    return corrected_angle



""" IMPLEMENTATION AND COMPARISONS OF DIFFERENT MODELS & ALGORITHMS """

""" Calculation of maximal angle due to backtracking """

# Calculate astronomical tracking angles with and without backtracking

astronomical_tracking_angles_no_backtracking = pvlib.tracking.singleaxis(
    apparent_zenith=apparent_zenith,
    apparent_azimuth=azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=False, 
    gcr=GCR) 

astronomical_tracking_angles = pvlib.tracking.singleaxis(
    apparent_zenith=apparent_zenith,
    apparent_azimuth=azimuth,
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=True, 
    gcr=GCR)

# Night time (no tracker rotation calculated) is set to 0°

astronomical_tracking_angles_no_backtracking['tracker_theta'] = astronomical_tracking_angles_no_backtracking['tracker_theta'].fillna(0)
astronomical_tracking_angles['tracker_theta'] = astronomical_tracking_angles['tracker_theta'].fillna(0)

# Creation of a "tracking" DataFrame from the astronomical angles

astronomical_tracking = pd.DataFrame(data=None, index=astronomical_tracking_angles.index)

# Convert rotation angles (floats) to the closest integer (in float type): e.g. 14.3 becomes 14.0

astronomical_tracking['angle'] = astronomical_tracking_angles['tracker_theta'].round(0) 

# Calculating the limit backtracking angle

max_angle_backtracking = find_max_angle_backtracking(astronomical_tracking_angles, astronomical_tracking_angles_no_backtracking, max_angle)

previous_angle = astronomical_tracking_angles['tracker_theta'].iloc[-1]

best_angle = model_brute_force_search(weather.iloc[-1], previous_angle, solpos.iloc[-1], max_angle_backtracking['angle'].iloc[-1], w_min, resolution)

optimal_angle = TeraBase(astronomical_tracking_angles['tracker_theta'].iloc[-1], best_angle, weather['Variability GHI'].iloc[-1], w_min, threshold_VI, resolution)
optimal_angle = optimal_angle.round(0)

'Forecast'

#FORECAST_brute_force_search_extended['angle'].plot(title='Tracking Curve', label="Forecast brute force search - limited tracker speed", ax=axes[0], color='pink')
#FORECAST_CENER_extended['angle'].plot(title='Tracking Curve', label="Forecast CENER",ax=axes[0], color='red')
#FORECAST_binary_mode_extended['angle'].plot(title='Tracking Curve', label="Forecast binary mode - either astronomical angle or 0",ax=axes[0], color='black')

#FORECAST_brute_force_search_extended['POA global'].plot(title='Irradiance', label="POA Forecast brute force search - limited tracker speed", ax=axes[1], color='pink')
#FORECAST_CENER_extended['POA global'].plot(title='Irradiance', label="POA Forecast CENER", ax=axes[1], color='red')
#FORECAST_binary_mode_extended['POA global'].plot(title='Irradiance', label="POA Forecast binary mode", ax=axes[1], color='black')

'Weather data'

weather['GHI'].plot(title='Irradiance', label="GHI", ax=axes[2], color='blue')
weather['DHI'].plot(title='Irradiance', label="DHI", ax=axes[2], color='orange')
weather['DNI'].plot(title='Irradiance', label="DNI", ax=axes[2], color='green')
ax2 = axes[2].twinx()
weather['Variability GHI'].plot(title='Irradiance', label="Variability", ax=ax2, color='black')

axes[0].legend(title="Tracker Tilt")
axes[1].legend(title="POA Irradiance")
axes[2].legend(title="Irradiance")
axes[2].set_ylabel("Irradiance (W/m2)")
ax2.set_ylabel("Variability factor")

axes[0].grid()
axes[1].grid()
axes[2].grid()

plt.legend()
plt.show()

