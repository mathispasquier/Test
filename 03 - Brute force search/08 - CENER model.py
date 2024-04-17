# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

Different optimal angle models & model implementation algorithm are compared

    Reference models:
        astronomical_tracking: astronomical tracking angles (commonly used)
        brute_force_search_infinite_speed: [1 min resolution] brute force search to find the best rotation angle, with no constraints for the tracker movement. It is assumed to be the best theoretical model (technically not feasible)

    With forecast data:
        brute_force_search : 
"""

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

from math import cos, sin, atan, radians, degrees, floor

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
max_angle = 55 # Maximum angle for the tracker rotation, in degrees
U = 24 # Tracker voltage, in V
I = 2 # Tracker current, in A
w_min = 9 # Tracker rotation speed, in degrees/minute
eta = 0.4 # Hesitation factor for the TeraBase model

# Data source: the file should contain 1 column with datetimes, 1 column with 'GHI' measurements and (if applicable) 1 column with 'DHI' measurements and 1 column with 'DNI' measurements

filename = '../../Risø Data formatted_2.csv' # Name and path for the data file
filedate_beginning = '2023-01-01' # Beginning date in the data file
filedate_end = '2024-01-01' # End date in the data file
dataresolution = '1min' # Time step in the data file (or data frequency to use)

# Simulation parameters

begin = '2023-06-01 00:00:00' # Beginning date for the simulation
end = '2023-06-02 00:00:00' # End date for the simulation
resolution = 5 # Tracker optimization resolution in minutes (if needed)
decomposition = False # Whether a decomposition model should be used or not
clearskymodel = False # Whether a clear sky model should be used or not

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
    weather['DNI'] = clearsky['DNI'] 

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

# Night time (no tracker rotation calculating) is set to 0°

astronomical_tracking_angles_no_backtracking['tracker_theta'] = astronomical_tracking_angles_no_backtracking['tracker_theta'].fillna(0)
astronomical_tracking_angles['tracker_theta'] = astronomical_tracking_angles['tracker_theta'].fillna(0)

# Calculating the limit backtracking angle

max_angle_backtracking = find_max_angle_backtracking(astronomical_tracking_angles, astronomical_tracking_angles_no_backtracking, max_angle)

""" Resample data using the relevant resolution """

resamp = str(resolution)+'min' # Tracker optimization resolution in minutes, if needed
weather_resamp = weather.resample(resamp).first()
solpos_resamp = solpos.resample(resamp).first()
weather_resamp = weather.resample(resamp).first()

max_angle_backtracking_resamp = max_angle_backtracking.resample(resamp).first()

""" Optimal angle models: brute force search, CENER, and binary mode """

def model_brute_force_search(weather, solar_position, max_angle_backtracking, w_min, resolution=1):
    """
    Calculates the optimized tracker rotation angle using the brute force search model. 
    For all the possible angles of rotation, a transposition model (e.g. Perez model) calculates the associated POA irradiance. 
    The angle that maximizes the POA irradiance is the optimized tracker rotation angle.
    From its previous position, the tracker can only move by +/- its rotation speed*resolution.
    
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
    
    brute_force_search = pd.DataFrame(data=None, index=weather.index)
    brute_force_search['angle'] = 0
    optimal_angle = 0
    previous_angle = 0

    for i in range(weather.index.size):
        
        max_irradiance = 0
        optimal_angle = 0
        
        min_angle_range = max(-max_angle_backtracking['angle'].iloc[i],previous_angle - w_min*resolution)
        max_angle_range = min(max_angle_backtracking['angle'].iloc[i],previous_angle + w_min*resolution)
        
        for angle in range(min_angle_range,max_angle_range,1):
            
            if angle < 0:
                surface_azimuth = 90 
                angle_abs = abs(angle)
            else:
                surface_azimuth = 270
                angle_abs = angle
            
            if angle == 0:
                total_irradiance = weather['GHI'].iloc[i]
            else:
                total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                                     surface_azimuth=surface_azimuth, 
                                                                     dni=weather['DNI'].iloc[i], 
                                                                     ghi=weather['GHI'].iloc[i], 
                                                                     dhi=weather['DHI'].iloc[i], 
                                                                     solar_zenith=solar_position['apparent_zenith'].iloc[i], 
                                                                     solar_azimuth=solar_position['azimuth'].iloc[i],
                                                                     dni_extra=weather['DNI_extra'].iloc[i],
                                                                     airmass=weather['airmass'].iloc[i],
                                                                     model='perez',
                                                                     model_perez='allsitescomposite1990') 
                total_irradiance = total_irrad['poa_global']
                       
            
            if total_irradiance > max_irradiance:
                max_irradiance = total_irradiance
                optimal_angle = angle
        
        brute_force_search['angle'].iloc[i] = optimal_angle
        previous_angle = optimal_angle
    
    return brute_force_search

def model_CENER(weather, solar_position, axis_azimuth, albedo, max_angle_backtracking):
    """
    Calculates the optimized tracker rotation angle using the CENER model.
    The CENER model uses the isotropic transposition model to provide an analytical equation for the optimized tracker rotation angle.

    Parameters
    ----------
    weather : DataFrame(index: DatetimeIndex, GHI: float, DHI: float, DNI: float, DNI_extra: float, airmass: float)
        Weather DataFrame that contains irradiance data (GHI, DHI, DNI, DNI extraterrestrial) and air mass.
    solar_position : DataFrame(index: DatetimeIndex, apparent_zenith: float, zenith: float, apparent_elevation: float, elevation: float, azimuth: float, equation_of_time: float)
        DataFrame that contains the parameters of the solar position (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time).
    axis_azimuth : int
        Tracker axis azimuth.
    albedo : float
        Ground albedo.
    max_angle_backtracking : DataFrame(index: DatetimeIndex, angle: int
        Floor value (absolute value) of the limit angle for the tracker rotation angle.

    Returns
    -------
    CENER : DataFrame(index: DatetimeIndex, angle: int)
        Optimized tracker rotation angle calculated using the CENER method.

    """
    
    CENER = pd.DataFrame(data=None, index=weather.index)
    CENER['angle'] = 0
    
    for i in range(weather.index.size):
        
        if solar_position['azimuth'].iloc[i] < 180:
            surface_azimuth = 90
        else:
            surface_azimuth = 270
        
        num = weather['DNI'].iloc[i]*sin(radians(solar_position['apparent_zenith'].iloc[i]))*cos(radians(solar_position['azimuth'].iloc[i]-surface_azimuth))
        den = (weather['DHI'].iloc[i]-weather['GHI'].iloc[i]*albedo)/2+weather['DNI'].iloc[i]*cos(radians(solar_position['apparent_zenith'].iloc[i]))
        
        # Set optimal angle to 0 during nighttime
        if den == 0:
            optimal_angle = 0.0
        else:
            optimal_angle = atan(num/den)
        
        # Convert optimal angle to an integer, in degrees
        optimal_angle = degrees(optimal_angle)
        optimal_angle = floor(min(optimal_angle,max_angle_backtracking['angle'].iloc[i]))
        
        if solar_position['azimuth'].iloc[i] < 180:
            optimal_angle = -optimal_angle
        
        CENER['angle'].iloc[i] = optimal_angle
        #CENER['angle'] = CENER['angle'].fillna(0)
    
    return CENER
    
""" Other metrics: degrees moved and POA irradiance """

def calculate_degrees_moved(tracking):
    """
    Calculates the rotation (in degrees) for each timestep, from the previous position.
    Procedure that adds the rotation to the tracking DataFrame and returns nothing.

    Parameters
    ----------
    tracking : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model.

    Returns
    -------
    None.

    """
    
    tracking['degrees_moved'] = 0
    
    previous_angle = 0
    
    for i in range(tracking.index.size):
        
        angle = tracking['angle'].iloc[i]
        tracking.loc[tracking.index[i],'degrees_moved'] = abs(angle-previous_angle)
        previous_angle = angle

def model_binary_mode(astronomical_tracking, weather, solar_position):
    """
    Calculates the optimized tracker rotation angle using the binary model.
    Binary model takes either the astronomical tracker angle or the horizontal position (0°), whatever yields the most.
    For the astronomical tracking, the POA irradiance is calculated using a transposition model (e.g. Perez model)

    Parameters
    ----------
    astronomical_tracking : DataFrame(index: DatetimeIndex, angle: int)
        Astronomical angle calculated (backtracking is implemented). The angle is the floor value of the theoretical astronomical angle.
    weather : DataFrame(index: DatetimeIndex, GHI: float, DHI: float, DNI: float, DNI_extra: float, airmass: float)
        Weather DataFrame that contains irradiance data (GHI, DHI, DNI, DNI extraterrestrial) and air mass.
    solar_position : DataFrame(index: DatetimeIndex, apparent_zenith: float, zenith: float, apparent_elevation: float, elevation: float, azimuth: float, equation_of_time: float)
        DataFrame that contains the parameters of the solar position (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time).

    Returns
    -------
    binary_mode : DataFrame(index: DatetimeIndex, angle: int)
        Optimized tracker rotation angle calculated using the binary model.

    """
    
    binary_mode = pd.DataFrame(data=None, index=weather.index)
    binary_mode['angle'] = 0
    
    astronomical_tracking_copy = astronomical_tracking.copy()
    calculate_POA_transposition(astronomical_tracking_copy, weather, solar_position)
    
    for i in range(weather.index.size):
        
        if weather['GHI'].iloc[i] < astronomical_tracking_copy['POA global'].iloc[i] :
            binary_mode['angle'].iloc[i] = astronomical_tracking_copy['angle'].iloc[i]
    
    return binary_mode

def calculate_POA_transposition(tracking, weather, solar_position):
    """
    Calculates the POA irradiance using a transposition model (e.g. Perez model) for each tracker rotation angle.
    Procedure that adds the POA irradiance to the tracking DataFrame and returns nothing.

    Parameters
    ----------
    tracking : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model.
    weather : DataFrame(index: DatetimeIndex, GHI: float, DHI: float, DNI: float, DNI_extra: float, airmass: float)
        Weather DataFrame that contains irradiance data (GHI, DHI, DNI, DNI extraterrestrial) and air mass.
    solar_position : DataFrame(index: DatetimeIndex, apparent_zenith: float, zenith: float, apparent_elevation: float, elevation: float, azimuth: float, equation_of_time: float)
        DataFrame that contains the parameters of the solar position (apparent zenith, zenith, apparent elevation, elevation, azimuth, equation of time).

    Returns
    -------
    None.

    """
    
    tracking['POA global'] = 0.0
    
    for i in range(tracking.index.size):
        
        angle = tracking['angle'].iloc[i]
            
        if angle < 0:
            surface_azimuth = 90 
            angle_abs = abs(angle)
        else:
            surface_azimuth = 270
            angle_abs = angle
        
        if angle == 0:
            tracking.loc[tracking.index[i],'POA global'] = weather['GHI'].iloc[i]
        else:
            total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt=angle_abs, 
                                                                surface_azimuth=surface_azimuth, 
                                                                dni=weather['DNI'].iloc[i], 
                                                                ghi=weather['GHI'].iloc[i], 
                                                                dhi=weather['DHI'].iloc[i], 
                                                                solar_zenith=solar_position['apparent_zenith'].iloc[i], 
                                                                solar_azimuth=solar_position['azimuth'].iloc[i],
                                                                dni_extra=weather['DNI_extra'].iloc[i],
                                                                airmass=weather['airmass'].iloc[i],
                                                                model='perez',
                                                                model_perez='allsitescomposite1990')
            tracking.loc[tracking.index[i], 'POA global'] = total_irrad['poa_global']

""" Computing for the different models """

# Calculation of the optimal tracker rotation angles with no constraints on the tracker movement (i.e. infinite tracker speed) using brute force search

brute_force_search_infinite_speed = model_brute_force_search(weather, solpos, max_angle_backtracking, 2*max_angle)
calculate_POA_transposition(brute_force_search_infinite_speed, weather, solpos)
calculate_degrees_moved(brute_force_search_infinite_speed)

# Calculation of the optimal tracker rotation angles bound by the tracker rotation speed using brute force search

brute_force_search = model_brute_force_search(weather, solpos, max_angle_backtracking, w_min)
calculate_POA_transposition(brute_force_search, weather, solpos)
calculate_degrees_moved(brute_force_search)

# Calculation of the optimal tracker rotation angles using the CENER model

CENER = model_CENER(weather, solpos, axis_azimuth, albedo, max_angle_backtracking)
calculate_POA_transposition(CENER, weather, solpos)
calculate_degrees_moved(CENER)

""" Extending the resampled brute force search tracking angles to the whole time index """
            
def extend_rotation_angle_forecast(tracking_resamp, resolution, index):  
    
    """
    Returns the tracker rotation angle every minute using the optimal rotation angle calculated for another resolution (e.g. 15 min).
    The tracker is moved at a constant rotation speed towards the next optimal position until it reaches this position, after what it remains stable.

    Parameters
    ----------
    tracking_resamp : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model at a given resolution (e.g. every 15 min).
    resolution : int, optional
        Tracker optimization resolution in minutes. The default is 1.
     index : DatetimeIndex
         Index of the period with a 1 minute resolution

    Returns
    -------
    tracking_resamp : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model with a 1 min resolution.

    """ 
    
    tracking = pd.DataFrame(data=None, index=index)
    tracking['angle'] = 0
    
    for i in range(tracking.index.size):
        
        # Check if this specific time corresponds to an optimal angle calculated in the resampled tracking
        if i%resolution == 0:
            tracking['angle'].iloc[i] = tracking_resamp['angle'].iloc[i//resolution]
        # Check if there is no optimal position calculated later: in that case, do not move the tracker
        elif i//resolution + 1 >  tracking_resamp.index.size - 1:
            tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1]
        # Otherwise, move the tracker at its rotation speed until the next optimal position is reached
        else:
            next_angle = tracking_resamp['angle'].iloc[i//resolution + 1]
            
            # Calculate the rotation (in degrees) to reach the next optimal angle
            delta_theta = next_angle-tracking['angle'].iloc[i-1]
            
            if delta_theta == 0:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1]
            elif delta_theta < 0:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1] + max(-w_min, delta_theta)
            else:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1] + min(w_min, delta_theta)

    return tracking

def extend_rotation_angle_real_time(tracking_resamp, resolution, index):  
    
    """
    Returns the tracker rotation angle every minute using the optimal rotation angle calculated for another resolution (e.g. 15 min).
    The tracker is moved at a constant rotation speed to the current optimal position until it reaches this position, after what it remains stable.

    Parameters
    ----------
    tracking_resamp : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model at a given resolution (e.g. every 15 min).
    resolution : int, optional
        Tracker optimization resolution in minutes. The default is 1.
     index : DatetimeIndex
         Index of the period with a 1 minute resolution

    Returns
    -------
    tracking_resamp : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model with a 1 min resolution.

    """ 
    
    tracking = pd.DataFrame(data=None, index=index)
    tracking['angle'] = 0
    
    for i in range(tracking.index.size):
        
        # The tracker starts at the horizontal position
        if i == 0:
            None
        else:
            optimal_angle = tracking_resamp['angle'].iloc[i//resolution]
            
            # Calculate the rotation (in degrees) to reach the next optimal angle
            delta_theta = optimal_angle-tracking['angle'].iloc[i-1]
            
            if delta_theta == 0:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1]
            elif delta_theta < 0:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1] + max(-w_min, delta_theta)
            else:
                tracking['angle'].iloc[i] = tracking['angle'].iloc[i-1] + min(w_min, delta_theta)

    return tracking

# Calculation of the optimal tracker rotation angles at a given resolution (e.g. 15 min) using brute force search

brute_force_search_resamp = model_brute_force_search(weather_resamp, solpos_resamp, max_angle_backtracking_resamp, w_min, resolution)
calculate_POA_transposition(brute_force_search_resamp, weather_resamp, solpos_resamp)
calculate_degrees_moved(brute_force_search_resamp)

# From the brute force search at a given resolution (e.g. 15 min), the position of the tracker is calculated every minute using forecast data (the next position is assumed to be known)

brute_force_search_extended_forecast = extend_rotation_angle_forecast(brute_force_search_resamp, resolution, weather.index)
calculate_POA_transposition(brute_force_search_extended_forecast, weather, solpos)
calculate_degrees_moved(brute_force_search_extended_forecast)

# From the brute force search at a given resolution (e.g. 15 min), the position of the tracker is calculated every minute for a real-time application (the next position is assumed to be unknown)

brute_force_search_extended_real_time = extend_rotation_angle_real_time(brute_force_search_resamp, resolution, weather.index)
calculate_POA_transposition(brute_force_search_extended_real_time, weather, solpos)
calculate_degrees_moved(brute_force_search_extended_real_time)

""" Comparison with astronomical tracking """

# Creation of a "tracking" DataFrame from the astronomical angles

astronomical_tracking = pd.DataFrame(data=None, index=astronomical_tracking_angles.index)

# Convert rotation angles (floats) to the closest integer (in float type): e.g. 14.3 becomes 14.0

astronomical_tracking['angle'] = astronomical_tracking_angles['tracker_theta'].round(0) 

# Calculate POA irradiance & degrees moved for astronomical tracking

calculate_POA_transposition(astronomical_tracking, weather, solpos)
calculate_degrees_moved(astronomical_tracking)

""" Simulations usinf the for the binary model """

# Calculation of the optimal tracker rotation angles using the binary model

binary_mode = model_binary_mode(astronomical_tracking, weather, solpos)
calculate_POA_transposition(binary_mode, weather, solpos)
calculate_degrees_moved(binary_mode)

""" Implementing the TeraBase model """

def TeraBase(astronomical_tracking, tracking, hesitation_factor, w_min, resolution=1):
    """
    The TeraBase model calculates a corrected tracker rotation angle between the astronomical angle and the optimal tracker rotation angle (calculated with a model).
    The calculations takes into account a movement penalty, which represeOnts the amount of time a tracker must spend in transit from the previous rotation angle to the current rotation angle and is a percentage of the total amount of time encompassed by that timestamp.
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
    resolution : int, optional
        Tracker optimization resolution in minutes. The default is 1.

    Returns
    -------
    TeraBase : DataFrame(index: DatetimeIndex, angle: int)
        Tracker rotation angle calculated using an optimal angle model and then corrected with the TeraBase model.

    """
    TeraBase = pd.DataFrame(data=None, index=tracking.index)
    TeraBase['Movement penalty'] =  0.0
    TeraBase['angle'] =  0
    
    for i in range(TeraBase.index.size):
        
        movement_penalty = abs(astronomical_tracking['angle'].iloc[i]-tracking['angle'].iloc[i])/(w_min*resolution) # w_min is the tracker rotation speed (in min.) and resolution is the number of minutes in the timestep
        TeraBase.loc[TeraBase.index[i],'Movement penalty'] =  movement_penalty
        
        # The maximum hesitation factor is 1-movement_penalty
        hesitation_factor_parameter = min(hesitation_factor,1-movement_penalty)
        
        corrected_angle = ((1-movement_penalty/100-hesitation_factor_parameter)*tracking['angle'].iloc[i])+((movement_penalty/100)*0.5*(astronomical_tracking['angle'].iloc[i]+tracking['angle'].iloc[i]))+(hesitation_factor_parameter*astronomical_tracking['angle'].iloc[i])
        TeraBase.loc[TeraBase.index[i],'angle'] =  floor(corrected_angle)
    
    return TeraBase

# Calculation of the rotation angle using the TeraBase model

brute_force_search_TeraBase = TeraBase(astronomical_tracking, brute_force_search, eta, w_min)
calculate_POA_transposition(brute_force_search_TeraBase, weather, solpos)
calculate_degrees_moved(brute_force_search_TeraBase)

""" Implementing TeraBase model with brute force search as ideal angle, using theta_i(t) and theta_s(t+5) to calculate theta_c(t+5) """

# Calculation of the astronomical angle with an offset of [resolution]

solpos_offset = pvlib.solarposition.get_solarposition(weather.index+pd.Timedelta(resamp), latitude, longitude, altitude)
astronomical_tracking_angles_offset = pvlib.tracking.singleaxis(
    apparent_zenith=solpos_offset['apparent_zenith'],
    apparent_azimuth=solpos_offset['azimuth'],
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=True, 
    gcr=GCR)

# Creation of a "tracking" DataFrame from the astronomical angles with an offset of [resolution]

astronomical_tracking_offset = pd.DataFrame(data=None, index=astronomical_tracking_angles_offset.index)
astronomical_tracking_offset['angle'] = astronomical_tracking_angles_offset['tracker_theta'].fillna(0).round(0) # Convert rotation angles (floats) to the closest integer (in float type): e.g. 14.3 becomes 14.0

# Resampling of the astronomical tracking with an offset to correspond to the optimal tracker rotation angle calculated at [resolution]

astronomical_tracking_offset = astronomical_tracking_offset.resample(resamp).first()

# Calculation of the corrected tracker angle using the TeraBase model at [resolution]

brute_force_search_TeraBase_offset = TeraBase(astronomical_tracking_offset, brute_force_search_infinite_speed.resample(resamp).first(), eta, w_min, resolution)
brute_force_search_TeraBase_offset = extend_rotation_angle_forecast(brute_force_search_TeraBase_offset, resolution, weather.index)
calculate_POA_transposition(brute_force_search_TeraBase_offset, weather, solpos)
calculate_degrees_moved(brute_force_search_TeraBase_offset)

""" Implementation of an electrical model """

"""module_parameters = {'pdc0': 1, 'gamma_pdc': -0.004, 'b': 0.05}
temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
array = pvlib.pvsystem.Array(mount=mount, module_parameters=module_parameters,
                       temperature_model_parameters=temp_params)
system = pvsystem.PVSystem(arrays=[array], inverter_parameters={'pdc0': 1})
mc = modelchain.ModelChain(system, loc, spectral_model='no_loss')"""

""" Calculation of KPIs for tracking """

def KPIs(tracking, w_min):
    """
    Calculates KPIs for the optimal tracking model.
    The tracking angles must be provided with a 1 min resolution

    Parameters
    ----------
    tracking : DataFrame(index: DatetimeIndex, angle: int)
        Optimal tracker rotation angle calculated using an optimal angle model.
    w_min : int or float
        Tracker rotation speed, in minutes.

    Returns
    -------
    total_degrees_moved : int
        Sum of degrees moved by the tracker.
    delta_t_moved : float
        Total time (in h) during which the tracker moved.
    consumption_tracker : float
        Energy consumption of the tracker in kWh.
    insolation : float
        Total POA insolation in kWh.

    """
    """ 
    Calculate KPIs based on the rotation angles & POA data
    """
    
    total_degrees_moved = tracking['degrees_moved'].sum() 
    delta_t_moved = total_degrees_moved/(w_min*60)
    consumption_tracker = U*I*delta_t_moved/1000
    insolation = (tracking['POA global'].mean()/1000)*(tracking.index.size/60)
    
    return (total_degrees_moved, delta_t_moved, consumption_tracker, insolation)

KPIs_brute_force_search = KPIs(brute_force_search, w_min)
KPIs_brute_force_search_infinite_speed = KPIs(brute_force_search_infinite_speed, w_min)
KPIs_brute_force_search_extended_forecast = KPIs(brute_force_search_extended_forecast, w_min)
KPIs_brute_force_search_extended_real_time = KPIs(brute_force_search_extended_real_time, w_min)
KPIs_brute_force_search_TeraBase = KPIs(brute_force_search_TeraBase, w_min)
KPIs_astronomical = KPIs(astronomical_tracking, w_min)
KPIs_brute_force_search_TeraBase_offset = KPIs(brute_force_search_TeraBase_offset, w_min)

""" Plot data """

# Tracking curves & POA irradiance

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

#brute_force_search['angle'].plot(title='Tracking Curve', label="Brute force search - limited tracker speed", ax=axes[0])
brute_force_search_infinite_speed['angle'].plot(title='Tracking Curve', label="Brute force search - infinite tracker speed", ax=axes[0])
#brute_force_search_TeraBase['angle'].plot(title='Tracking Curve', label="Brute force search - TeraBase model", ax=axes[0])
#brute_force_search_extended_forecast['angle'].plot(title='Tracking Curve', label="Brute force search - extended from another resolution", ax=axes[0])
astronomical_tracking['angle'].plot(title='Tracking Curve', label="Astronomical tracking",ax=axes[0])
#CENER['angle'].plot(title='Tracking Curve', label="CENER",ax=axes[0])
#binary_mode['angle'].plot(title='Tracking Curve', label="Binary mode - either astronomical angle or 0",ax=axes[0])
brute_force_search_TeraBase_offset['angle'].plot(title='Tracking Curve', label="Brute force search - TeraBase model real-time",ax=axes[0])

#brute_force_search['POA global'].plot(title='Irradiance', label="POA brute force search - limited tracker speed", ax=axes[1])
brute_force_search_infinite_speed['POA global'].plot(title='Irradiance', label="POA brute force search - infinite tracker speed", ax=axes[1])
#brute_force_search_TeraBase['POA global'].plot(title='Irradiance', label="POA brute force search - TeraBase model", ax=axes[1])
#brute_force_search_extended_forecast['POA global'].plot(title='Irradiance', label="POA brute force search - extended from another resolution", ax=axes[1])
astronomical_tracking['POA global'].plot(title='Irradiance', label="POA astronomical tracking", ax=axes[1])
#CENER['POA global'].plot(title='Irradiance', label="POA CENER", ax=axes[1])
#binary_mode['POA global'].plot(title='Irradiance', label="POA binary mode", ax=axes[1])
brute_force_search_TeraBase_offset['POA global'].plot(title='Irradiance', label="POA brute force search - TeraBase model real-time", ax=axes[1])

weather['GHI'].plot(title='Irradiance', label="GHI", ax=axes[2], color='blue')
weather['DHI'].plot(title='Irradiance', label="DHI", ax=axes[2], color='orange')
weather['DNI'].plot(title='Irradiance', label="DNI", ax=axes[2], color='green')

axes[0].legend(title="Tracker Tilt")
axes[1].legend(title="POA Irradiance")
axes[2].legend(title="Irradiance")

axes[0].grid()
axes[1].grid()
axes[2].grid()

plt.legend()
plt.show()

