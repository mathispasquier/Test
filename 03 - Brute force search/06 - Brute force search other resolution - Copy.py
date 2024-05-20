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

from math import cos, sin, acos, atan, atan2, radians, degrees, floor, ceil

""" Input parameters"""

# Site parameters

tz='Etc/GMT-1'
latitude = 55.696166
longitude = 12.105216
altitude = 14.5
GCR = 0.28
name = 'Risoe'

# Tracker & axis parameters

axis_tilt=0
axis_azimuth=180
max_angle=55 # Maximum angle for the tracker rotation
U = 24 # in V
I = 2 # in A
w_min = 9 # Tracker rotation speed, in minutes
w = w_min/60 # Tracker rotation speed, in degrees/second

# Data parameters

resolution = 15 # tracker optimization resolution in minutes, if needed

""" Get weather data (real data from .csv file) """

real_weather = pd.read_csv('../../Risø Data formatted_2.csv')

real_weather.index = pd.date_range('2023-01-01', '2024-01-01', freq='1min', tz=tz)
real_weather = real_weather.drop(['TmStamp'],axis=1)

""" Select the days when to calculate via brute force search """

begin = '2023-06-01 00:00:00'
end = '2023-06-02 00:00:00'

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

#linketurbidity = pvlib.clearsky.lookup_linke_turbidity(real_weather.index, latitude, longitude)

GHI = real_weather["GHI"] 
DHI = real_weather["DHI"]
DNI = real_weather["DNI"] 

""" Calculation of maximum angle due to backtracking """

def find_max_angle_backtracking(astronomical_tracking_angles_backtracking, astronomical_tracking_angles_no_backtracking, max_angle):

    """
    Returns the limit angle (absolute value): minimum between the backtracking angle and the maximum angle of rotation.
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

astronomical_tracking_angles_no_backtracking['tracker_theta'] = astronomical_tracking_angles_no_backtracking['tracker_theta'].fillna(0)
astronomical_tracking_angles['tracker_theta'] = astronomical_tracking_angles['tracker_theta'].fillna(0)
max_angle_backtracking = find_max_angle_backtracking(astronomical_tracking_angles, astronomical_tracking_angles_no_backtracking, max_angle)

""" Resample data using the relevant resolution """

resamp = str(resolution)+'min' # tracker optimization resolution in minutes, if needed
real_weather_resamp = real_weather.resample(resamp).first()
solpos_resamp = solpos.resample(resamp).first()
apparent_zenith_resamp = apparent_zenith.resample(resamp).first()
zenith_resamp = zenith.resample(resamp).first()
azimuth_resamp = azimuth.resample(resamp).first()
DNI_extra_resamp = DNI_extra.resample(resamp).first()
airmass_resamp = airmass.resample(resamp).first()
GHI_resamp = GHI.resample(resamp).first()
DHI_resamp = DHI.resample(resamp).first()
DNI_resamp = DNI.resample(resamp).first()

max_angle_backtracking_resamp = max_angle_backtracking.resample(resamp).first()

""" For each time, calculate optimal angle of rotation (with 1° resolution) that yields the maximum POA
Use a transposition model to calculate POA irradiance from GHI, DNI and DHI """

def find_optimal_rotation_angle(ghi, dhi, dni, dni_extra, airmass, solar_position, GCR, max_angle_backtracking, w_min, resolution=1):
    
    """
    Find the optimal rotation angle within given limits.
    """
    
    brute_force_search = pd.DataFrame(data=None, index=ghi.index)
    brute_force_search['angle'] = 0.0
    optimal_angle = 0
    previous_angle = 0

    for i in range(ghi.index.size):
        
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
        
        brute_force_search['angle'].iloc[i] = optimal_angle
        previous_angle = optimal_angle
    
    return brute_force_search

def calculate_degrees_moved(tracking, tracking_angle):
    
    """ 
    Calculate the degrees moved from the last position for each timestamp
    """
    
    tracking['degrees_moved'] = 0.0
    
    previous_angle = 0
    
    for i in range(tracking.index.size):
        
        angle = tracking_angle.iloc[i]
        tracking['degrees_moved'].iloc[i] = abs(angle-previous_angle)
        previous_angle = angle


def calculate_POA_transposition(tracking, tracking_angle):
    
    """
    Calculate the POA irradiance
    """
    
    tracking['POA global'] = 0.0
    
    for i in range(tracking.index.size):
        
        angle = tracking_angle.iloc[i]
            
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
        
        tracking['POA global'].iloc[i] = total_irrad["poa_global"]

brute_force_search = find_optimal_rotation_angle(GHI, DHI, DNI, DNI_extra, airmass, solpos, GCR, max_angle_backtracking, w_min, 1)
calculate_POA_transposition(brute_force_search, brute_force_search['angle'])
calculate_degrees_moved(brute_force_search, brute_force_search['angle'])

brute_force_search_resamp = find_optimal_rotation_angle(GHI_resamp, DHI_resamp, DNI_resamp, DNI_extra_resamp, airmass_resamp, solpos_resamp, GCR, max_angle_backtracking_resamp, w_min, resolution)
calculate_POA_transposition(brute_force_search_resamp, brute_force_search_resamp['angle'])
calculate_degrees_moved(brute_force_search_resamp, brute_force_search_resamp['angle'])

""" Extending the resampled brute force search tracking angles to the whole time index """

def extend_rotation_angle(brute_force_search_resamp, ghi, dhi, dni, dni_extra, airmass, solar_position, max_angle_backtracking, w_min, resolution):
    
    brute_force_search = pd.DataFrame(data=None, index=ghi.index)
    brute_force_search['angle'] = 0.0
    #brute_force_search['POA global'] = 0.0
    #brute_force_search['degrees_moved'] = 0.0
    previous_angle = 0
    
    for i in range(0,brute_force_search.index.size):
        
        # Flag to check whether optimization is needed
        need_optimization = False
        
        # Check if the optimal angle of rotation was already calculated in brute force search tracking resampled
        if i%resolution == 0:
            optimal_angle = int(brute_force_search_resamp['angle'].iloc[i//resolution])
        # If no optimal position calculated in the next minutes, keep the actual position
        elif i + resolution - i%resolution >= brute_force_search.index.size:
            optimal_angle = previous_angle
        # Else, find the optimal angle
        else:
            next_angle = brute_force_search_resamp['angle'].iloc[i//resolution+1]
            delta_angle = next_angle-previous_angle
            min_time_to_move = abs(delta_angle)/w_min
            # Check if the tracker is already in the next optimal position
            if delta_angle == 0:
                optimal_angle = previous_angle
            # Check if the tracker must be moved to the maximal to the maximal delta_angle to reach the next optimal position
            elif min_time_to_move == resolution-i%resolution:
                optimal_angle = np.sign(delta_angle)*w_min + previous_angle
            # Check if the tracker must be moved at least some degrees to reach the next optimal position
            elif ceil(min_time_to_move) == resolution-i%resolution:
                need_optimization = True
                if delta_angle >= 0:
                    min_angle_range = int(max(-max_angle_backtracking['angle'].iloc[i],previous_angle + delta_angle%w_min))
                    max_angle_range = min(max_angle_backtracking['angle'].iloc[i],previous_angle + w_min)
                else:
                    min_angle_range = max(-max_angle_backtracking['angle'].iloc[i],previous_angle - w_min)
                    max_angle_range = int(min(max_angle_backtracking['angle'].iloc[i],previous_angle - abs(delta_angle)%w_min))
            # Other wise find the optimal position (no constraint except that the tracker should move towars the next optimal position)
            else:
                need_optimization = True
                if delta_angle >= 0:
                    min_angle_range = max(-max_angle_backtracking['angle'].iloc[i],previous_angle)
                    max_angle_range = min(max_angle_backtracking['angle'].iloc[i],previous_angle + w_min)
                else:
                    min_angle_range = max(-max_angle_backtracking['angle'].iloc[i],previous_angle - w_min)
                    max_angle_range = min(max_angle_backtracking['angle'].iloc[i],previous_angle)
            
            if need_optimization:
                
                max_irradiance = 0
                optimal_angle = 0
                
                for angle in range(min_angle_range,max_angle_range,1):
                    
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

        brute_force_search['angle'].iloc[i] = optimal_angle
        #brute_force_search['degrees_moved'].iloc[i] = abs(optimal_angle-previous_angle)
        previous_angle = optimal_angle
    
    return brute_force_search
            
brute_force_search_extended = extend_rotation_angle(brute_force_search_resamp, GHI, DHI, DNI, DNI_extra, airmass, solpos, max_angle_backtracking, w_min, resolution)
calculate_POA_transposition(brute_force_search_extended, brute_force_search_extended['angle'])
calculate_degrees_moved(brute_force_search_extended, brute_force_search_extended['angle'])

""" Comparison with astronomical tracking """

astronomical_tracking = pd.DataFrame(data=None, index=astronomical_tracking_angles.index)
astronomical_tracking['angle'] = astronomical_tracking_angles['tracker_theta'].round(0) # Convert rotation angles (floats) to the closest integer (in float type): e.g. 14.3 becomes 14.0

""" Calculate POA irradiance & degrees moved for astronomical tracking """

calculate_POA_transposition(astronomical_tracking, astronomical_tracking['angle'])
calculate_degrees_moved(astronomical_tracking, astronomical_tracking['angle'])

""" Calculation of KPIs for tracking """

def KPIs(tracking, w, resolution):
    
    """ 
    Calculate KPIs based on the rotation angles & POA data
    """
    
    total_degrees_moved = tracking['degrees_moved'].sum() # Sum of degrees moved by the tracker
    delta_t_moved = total_degrees_moved/w # Total time (in s) during which the tracker moved
    consumption_tracker = U*I*delta_t_moved/3600000 # Energy consumption of the tracker in kWh
    energy_yield = (tracking['POA global'].mean()/1000)*(tracking.index.size*resolution/60) # Average POA irradiance in kW * number of hours
    
    return (total_degrees_moved, delta_t_moved, consumption_tracker, energy_yield)

KPIs_brute_force_search = KPIs(brute_force_search, w, resolution)
KPIs_brute_force_search_resamp = KPIs(brute_force_search_resamp, w, resolution)
KPIs_brute_force_search_extended = KPIs(brute_force_search_extended, w, resolution)
KPIs_astronomical = KPIs(astronomical_tracking, w, resolution)

""" Plot data """

# Tracking curves & POA irradiance

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

brute_force_search['angle'].plot(title='Tracking Curve', label="Optimal tracking", ax=axes[0])
brute_force_search_extended['angle'].plot(title='Tracking Curve', label="Optimal tracking extended", ax=axes[0])
#astronomical_tracking['angle'].plot(title='Tracking Curve', label="Astronomical tracking",ax=axes[0])

brute_force_search['POA global'].plot(title='Irradiance', label="POA brute force search tracking", ax=axes[1])
brute_force_search_extended['POA global'].plot(title='Irradiance', label="POA brute force search tracking extended", ax=axes[1])
#astronomical_tracking["POA global"].plot(title='Irradiance', label="POA astronomical tracking", ax=axes[1])

axes[0].legend(title="Tracker Tilt")
axes[1].legend(title="Irradiance")


plt.legend()
plt.show()
    
    