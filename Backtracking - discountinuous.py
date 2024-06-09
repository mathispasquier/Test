# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:14:05 2024

@author: mpa

Implementation of discountinous backtracking
"""

import pvlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from math import cos, sin, atan, atan2, radians, degrees, isnan

""" Input parameters"""

tz = 'Europe/Copenhagen'
latitude = 55.696166
longitude = 12.105216
name = 'Risoe'
axis_tilt=0
axis_azimuth=180
max_angle=55
altitude = 14.5
backtrack = True
GCR = 0.28
step = 6 #step in ° for discontinuous backtracking

""" New class for discountinous backtracking - time steps """

class DiscontinuousTrackerMount(pvlib.pvsystem.SingleAxisTrackerMount):
    # inherit from SingleAxisTrackerMount so that we get the
    # constructor and tracking attributes (axis_tilt etc) automatically

    def get_orientation(self, solar_zenith, solar_azimuth):
        # Different trackers update at different rates; in this example we'll
        # assume a relatively slow update interval of 15 minutes to make the
        # effect more visually apparent.
        zenith_subset = solar_zenith.resample('15min').first()
        azimuth_subset = solar_azimuth.resample('15min').first()

        tracking_data_15min = pvlib.tracking.singleaxis(
            zenith_subset, azimuth_subset,
            self.axis_tilt, self.axis_azimuth,
            self.max_angle, self.backtrack,
            self.gcr, self.cross_axis_tilt
        )
        # propagate the 15-minute positions to 1-minute stair-stepped values:
        tracking_data_1min = tracking_data_15min.reindex(solar_zenith.index,
                                                         method='ffill')
        return tracking_data_1min

""" New class for discountinous backtracking - degree steps """

class DiscontinuousTrackerMountDegrees(pvlib.pvsystem.SingleAxisTrackerMount):
    # inherit from SingleAxisTrackerMount so that we get the
    # constructor and tracking attributes (axis_tilt etc) automatically

    def get_orientation(self, solar_zenith, solar_azimuth, step):
        # Different trackers update at different rates; in this example we'll
        # assume a relatively slow update interval of 15 minutes to make the
        # effect more visually apparent.
    
        tracking_data = pvlib.tracking.singleaxis(
            solar_zenith, solar_azimuth,
            self.axis_tilt, self.axis_azimuth,
            self.max_angle, self.backtrack,
            self.gcr, self.cross_axis_tilt
        )
        
        # Function implemented so the tracker finishes at 0° at the end of the day
        tracking_data['tracker_theta'] = tracking_data['tracker_theta'].fillna(0)
        
        angle = 0
        
        for index, data in tracking_data.iterrows():          
            tracker_theta = data["tracker_theta"]   
            if abs(tracker_theta - angle) < step:
                #if tracker_theta != 0:
                    tracking_data = tracking_data.drop(index)
            else:   
                angle += np.sign(tracker_theta-angle) * step
        
        tracking_data_degrees = tracking_data.reindex(solar_zenith.index,
                                                         method='ffill')

        return tracking_data_degrees


""" Calculation of the rotation angle - Based on the new class created """

fig, ax = plt.subplots()

times = pd.date_range('2019-01-01', '2019-01-02', freq='1min', tz=tz)

solpos = pvlib.solarposition.get_solarposition(times, 
                                               latitude, 
                                               longitude, 
                                               altitude)

mount = DiscontinuousTrackerMountDegrees(axis_tilt,
                                         axis_azimuth,
                                         max_angle,
                                         backtrack,
                                         GCR)

mount_time = DiscontinuousTrackerMount(axis_tilt,
                                         axis_azimuth,
                                         max_angle,
                                         backtrack,
                                         GCR)

apparent_zenith = solpos["apparent_zenith"]
azimuth = solpos["azimuth"]

backtracking_angles_discontinuous = mount.get_orientation(apparent_zenith, 
                                     azimuth,
                                     step)

backtracking_position_discontinuous = backtracking_angles_discontinuous['tracker_theta'].fillna(0)

backtracking_angles_time = mount_time.get_orientation(apparent_zenith, 
                                     azimuth)

backtracking_position_time = backtracking_angles_time['tracker_theta'].fillna(0)

""" Plot """

backtracking_position_time.plot(color='g')
plt.ylabel('Tracker Rotation [degree]')
plt.show()
