#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil 
import tensorflow as tf 
from igm.utils.math.interp1d_tf import interp1d_tf 
import xarray as xr

def initialize(cfg, state):

    # Check if an array of time-dependent temp and precip offset are provided by user (and initialize the variable to store it if yes)
    if cfg.processes.clim_station.climate_change_array == []:
        state.climatepar = None
    else:
        state.climatepar = np.array(cfg.processes.clim_station.climate_change_array[1:]).astype(np.float32)

    # Precompute the required constants (once) based on user-given climate parameters
    state.reference_precip_ice = cfg.processes.clim_station.reference_precipitation * 1.0989 # conversion from water equivalent to ice equivalent (using ice density of 910 kg m-3)
    state.precip_ref_change = ((cfg.processes.clim_station.precipitation_lapse_rate / 100) * state.reference_precip_ice) # equal to ~27.47 kg m-2 yr-1 ice equ. with default params. 

    produce_climate_data(cfg, state)

    state.air_temp = tf.Variable(state.air_temp_ref, dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_ref, dtype="float32")
    state.precipitation = tf.Variable(state.precipitation_ref, dtype="float32")

    # create the time loggers  
    state.tlast_clim_update = tf.Variable(-1.0e5000)
    

def update(cfg, state):
    """Update air temperature and precipitation based on modelled ice surface elevation changes."""

    # update climate fields each X years
    if (state.t - state.tlast_clim_update) >= cfg.processes.clim_station.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct climate at time : " + str(state.t.numpy())
            ) 

        # Explicitly reset state temp and precip variables to their initial values before applying changes (otherwise we can get a runaway effect...)
        state.air_temp.assign(state.air_temp_ref)  # Reset air temperature to initial reference temperature field
        state.precipitation.assign(state.precipitation_ref)  # Reset precipitation to initial reference precipitation field


        ### If set by user, apply the time-dependent climate changes
        if cfg.processes.clim_station.time_dependent_climate and state.climatepar is not None:
            # Interpolate temperature and precipitation offsets at current time
            temp_offset = interp1d_tf(state.climatepar[:, 0], state.climatepar[:, 1], state.t)
            temp_offset = tf.broadcast_to(temp_offset, tf.shape(state.air_temp))
            precip_offset = interp1d_tf(state.climatepar[:, 0], state.climatepar[:, 2], state.t)
            precip_offset = tf.broadcast_to(precip_offset, tf.shape(state.precipitation))

            # Apply temperature offsets (in °C) for current model time
            state.air_temp.assign_add(temp_offset)

            # Apply precipitation offset (given as percentage of original precip) for current model time
            state.precipitation.assign(state.precipitation * (precip_offset / 100.0))


        ### TEMP change due to ice surface elevation change
        # Compute surface elevation difference (how much usurf has changed from topg)
        delta_height = state.usurf - state.topg
        delta_height_inversed = -delta_height  # we inverse as surface Temp needs to decrease with increasing elevation offset from topg
        # Ensure delta_height_inversed has the same shape as state.air_temp
        delta_height_inversed = tf.broadcast_to(delta_height_inversed, tf.shape(state.air_temp))
        # Update air temperature using the user-defined adiabatic lapse rate
        state.air_temp.assign_add(cfg.processes.clim_station.adiabatic_lapse_rate * delta_height_inversed)

        ### PRECIP change due to ice surface elevation change
        new_diff_elev_precip = delta_height / 100
        # Ensure new_diff_elev_precip has the correct shape
        new_diff_elev_precip = tf.broadcast_to(new_diff_elev_precip, tf.shape(state.precipitation))
        # Update precipitation field
        delta_precip = new_diff_elev_precip * state.precip_ref_change
        new_precip = state.precipitation + delta_precip
        # Ensure precipitation does not fall below the minimum allowed (user defined)
        new_precip = tf.maximum(new_precip, cfg.processes.clim_station.min_precipitation_allowed)
        # Assign updated precipitation values
        state.precipitation.assign(new_precip)


        # update the time loggers
        state.tlast_clim_update.assign(state.t) 

        # Print climate data to make sure values make sense (optionnal)
        #tf.print("air_temp (°C): min =", tf.reduce_min(state.air_temp), "max =", tf.reduce_max(state.air_temp))
        #tf.print("air_temp_sd (°C): min =", tf.reduce_min(state.air_temp_sd), "max =", tf.reduce_max(state.air_temp_sd))
        #tf.print("precipitation (kg m-2 yr-1 ice equ.): min =", tf.reduce_min(state.precipitation), "max =", tf.reduce_max(state.precipitation))

        #tf.print("Shape of delta_height:", tf.shape(delta_height))
        #tf.print("Shape of air_temp:", tf.shape(state.air_temp))

        #tf.print("delta_height: min =", tf.reduce_min(delta_height), "max =", tf.reduce_max(delta_height))
        #tf.print("delta_precip: min =", tf.reduce_min(delta_precip), "max =", tf.reduce_max(delta_precip))

def finalize(cfg, state):
    pass



def produce_climate_data(cfg, state):

    # Precompute the required constants based on user-defined parameters
    # seconds_per_year = 365.25 * 24 * 3600  # Convert mm/yr to kg m^-2 s^-1


    # Create the climate data fields based on topg and user parameters

    # Compute the mean annual air temperature field using input topography
    diff_elev_temp = cfg.processes.clim_station.zero_degree_isotherm - state.topg
    air_temp = 0.0 + (cfg.processes.clim_station.adiabatic_lapse_rate * diff_elev_temp)

    # Compute the mean annual precipitation field using input topography
    diff_elev_precip = (state.topg - cfg.processes.clim_station.reference_precipitation_elevation) / 100
    precip_change = diff_elev_precip * state.precip_ref_change
    precipitation = state.reference_precip_ice + precip_change
    # Ensure precipitation is never below the minimum allowed
    precipitation = tf.maximum(precipitation, cfg.processes.clim_station.min_precipitation_allowed)

    # Create the space-invariant air_temp_sd variable
    air_temp_sd = tf.fill(tf.shape(state.thk), cfg.processes.clim_station.air_temperature_stdev)

    # Expand dimensions to add time (e.g., 12 months with data every months)
    # At the moment the climate data is constant accross the 12 months
    time_steps = 12  # Set this based on your needs (e.g., monthly data)
    air_temp = tf.expand_dims(air_temp, axis=0)  # Add time dimension
    air_temp = tf.repeat(air_temp, time_steps, axis=0)  # Repeat for all time steps

    air_temp_sd = tf.expand_dims(air_temp_sd, axis=0)
    air_temp_sd = tf.repeat(air_temp_sd, time_steps, axis=0)

    precipitation = tf.expand_dims(precipitation, axis=0)
    precipitation = tf.repeat(precipitation, time_steps, axis=0)

    # Apply seasonal variation with cosine yearly cycle if enabled (sinusoidal fluctuation around the mean temp using a cosine function.)
    if cfg.processes.clim_station.cosine_yearly_cycle_temp:
        months = np.arange(12)
        if cfg.processes.clim_station.southern_hemisphere_climate:
            seasonal_cycle = cfg.processes.clim_station.cosine_yearly_cycle_amplitude * np.cos(2 * np.pi * months / 12)
        else:
            seasonal_cycle = cfg.processes.clim_station.cosine_yearly_cycle_amplitude * np.cos(2 * np.pi * (months - 6) / 12)
        
        seasonal_cycle = tf.convert_to_tensor(seasonal_cycle, dtype=tf.float32)
        seasonal_cycle = tf.reshape(seasonal_cycle, (12, 1, 1))
        air_temp = air_temp + seasonal_cycle


    #######################   Save the climate as NetCDF (optional) #############
    if cfg.processes.clim_station.export_climate_ref:
        ds = xr.Dataset(
            {
                "air_temp": (["time", "y", "x"], air_temp.numpy()),
                "air_temp_sd": (["time", "y", "x"], air_temp_sd.numpy()),
                "precipitation": (["time", "y", "x"], precipitation.numpy()),
            },
            coords={
                "time": np.arange(12),  # 12 months
            },
        )
        # Save to NetCDF file
        ds.to_netcdf("climate_ref.nc")
    ################################################################################


    # We shift the climate time dimension to represent the hydrological year (start of the year becomes start of the accumulation season)
    # Start of accumulation season is set as 1st November in Northern Hemisphere, and 1st May in Southern hemisphere.
    shift = 2 / 12 if not cfg.processes.clim_station.southern_hemisphere_climate else 8 / 12
    # Apply shift using np.roll, this should work for any number of time steps in a year (monthly, weekly, daily data)
    air_temp = np.roll(air_temp.numpy(), int(len(air_temp) * (1 - shift)), axis=0)
    air_temp_sd = np.roll(air_temp_sd.numpy(), int(len(air_temp_sd) * (1 - shift)), axis=0)
    precipitation = np.roll(precipitation.numpy(), int(len(precipitation) * (1 - shift)), axis=0)


    
    state.air_temp_ref = tf.Variable(air_temp, dtype="float32")
    state.air_temp_sd_ref = tf.Variable(air_temp_sd, dtype="float32")
    state.precipitation_ref = tf.constant(precipitation, dtype="float32")
