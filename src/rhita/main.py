import rhita.data_processing as dp
import rhita.hazards_detection as hd
import rhita.output_options as oo
#import rhita.statistics as rstats
import numpy as np
#import pandas as pd
import os

def main(ds, ds_th, config):

    # 2. Time consistency (check if the time has always the same timestep, otherwise it raises an error) and conversion of time in a standard format 
    dp.check_time(ds, config)
    ds[config['data_structure']['time']] = dp.standard_time_format(ds, config)

    # 3. Data subsetting
    # spatial subsetting
    if config['subsetting'].getboolean('spatial_subset'):
        ds = dp.spatial_subsetting(ds, config, "data")
    if config['methods_parameters']['threshold1'] == 'map' or config['methods_parameters']['threshold1'] == 'map_time_of_year':
        ds_th = dp.spatial_subsetting(ds_th, config, "threshold")
    # temporal subsetting
    ds = dp.temporal_subsetting(ds, config)

    # 4. Extract the variables of interest
    var = ds[config['data_structure']['variable_name']].values
    lat = ds[config['data_structure']['y_coordinate']].values
    lon = ds[config['data_structure']['x_coordinate']].values
    lon_matrix, lat_matrix = np.meshgrid(lon, lat)
    time = ds[config['data_structure']['time']].values    
    ds.close()
    if config['methods_parameters']['threshold1'] == 'map':
        var_th = ds_th[config['data_structure']['variable_name_map']].values
        ds_th.close()
    elif config['methods_parameters']['threshold1'] == 'map_time_of_year':
        var_th = ds_th[config['data_structure']['variable_name_map']].values
        time_th = ds_th[config['data_structure']['time']].values
        ds_th.close()
    else:
        var_th = None

    # convert from K to C
    if config['data_structure']['UoM'] == "K":
        var = var - 273.15
        if var_th is not None:
            var_th = var_th - 273.15

    if config['data_structure']['UoM'] == "kg.m-2.s-1":
        var = var * 3600 * 24  # Convert kg.m-2.s-1 to mm/day

    # 5. Binarization of the data
    binary_map = hd.thresholding(var, var_th, time if config['methods_parameters']['threshold1'] == 'map_time_of_year' else None, time_th if config['methods_parameters']['threshold1'] == 'map_time_of_year' else None, config)

    # Compute the excess over the threshold
    if var_th is None:
        excess_var = var - float(config['methods_parameters']['fixed_th1'])
    else:
        excess_var = var - var_th 

    # 6. Identify the single events on the spatial domain
    events_2D = hd.events_2D(binary_map, lat_matrix, lon_matrix, config)

    # 7. Compute centroid for each 2D event
    centroids = hd.centroids(events_2D, var, excess_var, lat_matrix, lon_matrix, config)

    # 8. Identify the 3D event (time and space)
    network_3D, events_3D = hd.events_3D(centroids, events_2D, lat_matrix, lon_matrix, config)

    # 9. Save the results
    catalogue = config['output_options'].getboolean('catalogue')
    single_event_3D = config['output_options'].getboolean('single_event3D')
    single_event_tracking = config['output_options'].getboolean('single_event_tracking')
    oo.save_files(network_3D if single_event_tracking or catalogue else None,
                  events_3D if single_event_3D else None, 
                  var if single_event_3D else None, 
                  lat if single_event_3D else None, 
                  lon if single_event_3D else None,
                  time, 
                  config)
    
    with open(os.path.join(config['directories']['output'],config['methods_parameters']['hazard_type'],config['output_options']['folder_name_hazard'],'Config_file.ini'), 'w') as f:
        config.write(f)
