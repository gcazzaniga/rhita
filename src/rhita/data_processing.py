# This script contains functions for importing and subsetting data.

import os
import xarray as xr
import logging
import geopandas as gpd
import regionmask as rm
import numpy as np
import pandas as pd
from datetime import datetime
from rhita.utils import log_execution_time
import time as time
#import h5netcdf
from dask.distributed import Client

# Function for importing data and sorting in a standard order
@log_execution_time
def import_data(config, nc_files = None, data_dir = None):  
    x_coord_name = config['data_structure']['x_coordinate']
    y_coord_name = config['data_structure']['y_coordinate']
    time_name = config['data_structure']['time']
    parallel = config['settings'].getboolean('parallel')  
    data_dir = config['directories']['input_data'] if data_dir is None else data_dir
    # list netcdf file/s from the folder
    if nc_files is not None:
        files = nc_files
    else:
        files = os.listdir(data_dir)  
        nc_files = [file for file in files if file.endswith('.nc') or file.endswith('.nc4') or file.endswith('.nc3') or file.endswith('.HDF5')]
    if parallel:
        client = Client()
    # combine multiple files if necessary
    if len(nc_files) > 1:
        chunks = {'time': 300}
        ds = xr.open_mfdataset([os.path.join(data_dir, file) for file in nc_files], combine='by_coords', engine='h5netcdf', parallel = parallel if True else False, chunks = chunks, decode_cf=False)
    else:
        ds = xr.open_dataset(os.path.join(data_dir, nc_files[0]))   
    # sort the data by time, latitude, and longitude
    ds = ds.sortby([time_name])
    #if not ds[y_coord_name].values[0] < ds[y_coord_name].values[1]:
    #    ds = ds.sortby([y_coord_name])
    #if not ds[x_coord_name].values[0] < ds[x_coord_name].values[1]:
    #    ds = ds.sortby([x_coord_name])
    # transpose the data to have a standard format (time, lat, lon, other dimensions)
    other_dims = [dim for dim in ds.dims if dim not in [time_name, x_coord_name, y_coord_name]]
    transpose_dims = [time_name, y_coord_name, x_coord_name] + other_dims
    ds = ds.transpose(*transpose_dims)
    if parallel:
        client.close()
    logging.info("Data imported successfully")  
    print("Data imported successfully")
    return ds

# Function for importing the threshold map (when available)
@log_execution_time
def import_map_th(config):
    x_coord_name = config['data_structure']['x_coordinate_map']
    y_coord_name = config['data_structure']['y_coordinate_map']
    ds_th = xr.open_dataset(config['methods_parameters']['map_path'])
    if config['methods_parameters']['threshold1'] == 'map':
        # transpose the data to have a standard format (lat, lon, other dimensions)
        other_dims = [dim for dim in ds_th.dims if dim not in [x_coord_name, y_coord_name]]
        transpose_dims = [y_coord_name, x_coord_name] + other_dims
    elif config['methods_parameters']['threshold1'] == 'map_time_of_year':
        # transpose the data to have a standard format (time, lat, lon, other dimensions)
        time_name = config['data_structure']['time']
        other_dims = [dim for dim in ds_th.dims if dim not in [time_name, x_coord_name, y_coord_name]]
        transpose_dims = [time_name, y_coord_name, x_coord_name] + other_dims
    else:
        logging.error("Threshold map not recognized.")
    ds_th = ds_th.transpose(*transpose_dims)
    logging.info("Map of threshold imported successfully")  
    print("Map of threshold imported successfully")
    return ds_th

# Function for checking the time consistency of the data
@log_execution_time
def check_time(ds, config):
    if config['data_structure'].getboolean('nanoseconds'):
        datetime_format = [datetime.strptime(str(tt)[:-3], config['data_structure']['time_format_input']) for tt in ds[config['data_structure']['time']].values]
    elif config['data_structure']['time_format_input'] == 'numeric_hours':
        time_var = ds[config['data_structure']['time']]
        datetime_format = num2date(
            time_var[:],
            units=time_var.units,  # 'hours since the reference'
            calendar=time_var.calendar  # 'standard'
        )
    else:
        datetime_format = [datetime.strptime(str(tt), config['data_structure']['time_format_input']) for tt in ds[config['data_structure']['time']].values]
    if config['data_structure']['temporal_resolution'] == '1 month':
        deltas = pd.Series(datetime_format).diff().dropna().apply(lambda x: np.floor(x / np.timedelta64(1, 'W'))).values
    else:
        deltas = pd.Series(datetime_format).diff().dropna().values
    if not np.all(deltas == deltas[0]):
        logging.error('Missing time consistency.')
        raise ValueError("Missing time consistency.")
    
    logging.info("Time consistency check completed successfully")
    print("Time consistency check completed successfully")
    return

# Function for converting the time format to a standard format (ISO format)
@log_execution_time
def standard_time_format(ds, config):
    if config['data_structure'].getboolean('nanoseconds'):
        time_ISO = np.array([datetime.strptime(str(tt)[:-3], config['data_structure']['time_format_input']).strftime('%Y-%m-%dT%H-%M-%S') for tt in ds[config['data_structure']['time']].values])
    elif config['data_structure']['time_format_input'] == 'numeric_hours':
        time_var = ds[config['data_structure']['time']]
        time_ISO = np.array(num2date(
            time_var[:],
            units=time_var.units,  # 'hours since the reference'
            calendar=time_var.calendar  # 'standard'
        ))
        time_ISO = np.array([datetime.strptime(str(tt), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%H-%M-%S') for tt in time_ISO])
    else:
        time_ISO = np.array([datetime.strptime(str(tt), config['data_structure']['time_format_input']).strftime('%Y-%m-%dT%H-%M-%S') for tt in ds[config['data_structure']['time']].values])
    return time_ISO

# Function for spatial subsetting of data
@log_execution_time
def spatial_subsetting(ds, config):

    x_coord_name = config['data_structure']['x_coordinate']
    y_coord_name = config['data_structure']['y_coordinate']

    # mask the netcdf file with the shapefile
    if config['subsetting']['spatial_subset_method'] == 'shapefile':
        mask_file = os.path.join(config['directories']['mask'], config['subsetting']['mask_file'])
        sf = gpd.read_file(mask_file)
        lon_min, lat_min, lon_max, lat_max = sf.geometry.total_bounds
        if ds[x_coord_name].values[0] > ds[x_coord_name].values[1]:
            lon_min, lon_max = lon_max, lon_min
        if ds[y_coord_name].values[0] > ds[y_coord_name].values[1]:
            lat_min, lat_max = lat_max, lat_min
        ds_subset = ds.sel({x_coord_name: slice(lon_min, lon_max), y_coord_name: slice(lat_min, lat_max)}) 
        poly = rm.Regions(sf.geometry)
        mask = poly.mask(*(ds_subset[x_coord_name].values, ds_subset[y_coord_name].values))
        mask = mask.rename({'lon': x_coord_name, 'lat': y_coord_name})       
        ds = ds_subset.where(mask == 0)
    # mask the netcdf with a netcdf  
    elif config['subsetting']['spatial_subset_method'] == 'netcdf':
        mask_file = os.path.join(config['directories']['mask'], config['subsetting']['mask_file'])
        mask = xr.open_dataset(mask_file)
        # change the coordinates of the mask from 0 to 360 to -180 to 180
        # mask['lon'] = (mask['lon'] + 180) % 360 - 180
        mask_interp = mask.interp_like(ds)
        lon_min, lon_max = mask.lon.min().item(), mask.lon.max().item()
        lat_min, lat_max = mask.lat.min().item(), mask.lat.max().item()
        if ds[x_coord_name].values[0] > ds[x_coord_name].values[1]:
            lon_min, lon_max = lon_max, lon_min
        if ds[y_coord_name].values[0] > ds[y_coord_name].values[1]:
            lat_min, lat_max = lat_max, lat_min 
        mask_interp = mask_interp.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        ds_subset = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        ds_masked = ds_subset.copy()
        ds_masked[config['data_structure']['variable_name_map']] = ds_masked[config['data_structure']['variable_name_map']].where(mask_interp[config['subsetting']['var_name_mask']] < 60)
        ds = ds_masked
    # subset data based on coordinates
    elif config['subsetting']['spatial_subset_method'] == 'coordinates':
        lat_min = float(config['subsetting']['lat_min'])
        lon_min = float(config['subsetting']['lon_min'])
        lat_max = float(config['subsetting']['lat_max'])
        lon_max = float(config['subsetting']['lon_max'])
        if ds[x_coord_name].values[0] > ds[x_coord_name].values[1]:
            lon_min, lon_max = lon_max, lon_min
        if ds[y_coord_name].values[0] > ds[y_coord_name].values[1]:
            lat_min, lat_max = lat_max, lat_min
        ds = ds.sel({x_coord_name: slice(lon_min, lon_max), y_coord_name: slice(lat_min, lat_max)}) 
    else:
        logging.error("Spatial subsetting method not recognized.")
        raise ValueError("Spatial subsetting method not recognized.")
    
    if ds[y_coord_name].values[0] > ds[y_coord_name].values[1]:
        ds = ds.sortby([y_coord_name], ascending = True)
    if ds[x_coord_name].values[0] > ds[x_coord_name].values[1]:
        ds = ds.sortby([x_coord_name], ascending = True)

    logging.info("Spatial subsetting completed successfully")  
    print("Spatial subsetting completed successfully")
    return ds

# Function for temporal subsetting of data
@log_execution_time
def temporal_subsetting(ds, config):

    time_name = config['data_structure']['time']
    start_date = config['subsetting']['start_date']
    end_date = config['subsetting']['end_date']
    ds.chunk({time_name: -1})
    ds = ds.sel({time_name: slice(start_date, end_date)})
    
    logging.info("Temporal subsetting completed successfully")
    print("Temporal subsetting completed successfully")
    return ds