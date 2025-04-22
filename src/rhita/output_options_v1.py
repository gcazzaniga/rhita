# This script contains functions for saving the output files; depending on the users settings

import os
import logging
import numpy as np
import networkx as nx
import pandas as pd
import statistics as stat
import rhita.utils as utils
import xarray as xr
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
from datetime import datetime, timedelta
import time

def get_country_old(latitude, longitude):
    try:
        # Initialize Nominatim API
        geolocator = Nominatim(user_agent="my_geocoder_app_v1")
        # Get location details and specify the language as English
        location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
        # Ensure the location was found
        if location is None:
            return 'Location not found'
        # Extract address details safely
        address = location.raw.get('address', {})
        # Return the country name, or a fallback message if not found
        return address.get('country', 'Country not found')
    except GeocoderServiceError as e:
        # Handle errors such as service downtime or network issues
        return f"Error: {e}"
    except Exception as e:
        # Catch any other errors, such as invalid latitude/longitude
        return f"Unexpected error: {e}"

def get_country(latitude, longitude, retries=3, delay=1):
    geolocator = Nominatim(user_agent="my_geocoder_app_v1")
    
    for attempt in range(retries):
        try:
            # Perform the reverse geocoding with a longer timeout
            location = geolocator.reverse(f"{latitude}, {longitude}", language='en', timeout=5)
            
            # Check if location was found
            if location is None:
                return 'Location not found'
            
            # Extract and return the country from the address details
            address = location.raw.get('address', {})
            return address.get('country', 'Country not found')
        
        except GeocoderServiceError as e:
            # If it's the last attempt, return the error
            if attempt == retries - 1:
                return f"Error: {e}"
            time.sleep(delay)  # Wait before retrying
        
        except Exception as e:
            # Handle unexpected errors gracefully
            return f"Unexpected error: {e}"

    return "Failed after retries"

## Define a function to get the country name from latitude and longitude coordinates
#def get_country1(latitude, longitude):
#    # Initialize Nominatim API
#    geolocator = Nominatim(user_agent="my_geocoder_app_v1")
#    # Get location details and specify the language as English
#    location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
#    # Extract address details
#    address = location.raw['address']
#    # Return the country name
#    return address.get('country', 'Country not found')

def create_catalogue(network, time, config):

    # initialize an empty list to store the data
    data = []

    # Iterate over the single events network
    for _, subg in enumerate(network):

        # get the ID (minimum ID of all the events in the network)
        id = int(min(subg.nodes()))
        # Get the first date
        node_time = nx.get_node_attributes(subg,'time')
        date = time[int(min(node_time.values()))]
        # Get the list of impacted countries
        coord = list(nx.get_node_attributes(subg,'pos').values())
        # if subg is empty 
        if len(nx.get_node_attributes(subg,'countries').values()) == 0:
            countries = 'No country found'
        else:
            countries = np.unique(np.concatenate(list(nx.get_node_attributes(subg,'countries').values())))
        # Get the time distance 
        duration = max(nx.get_node_attributes(subg,'timestamp_from_start').values()) + 1
        # Get the volume
        volume = sum(nx.get_node_attributes(subg,'area').values())
        # Get the mean area
        area_mean = stat.mean(nx.get_node_attributes(subg,'area').values())
        # Get the max area
        area_max = max(nx.get_node_attributes(subg,'area').values())
        # Get mean severity
        severity_mean = stat.mean(nx.get_node_attributes(subg,'severity_mean').values())
        # Get integrated excess
        excess_sum = sum(nx.get_node_attributes(subg,'excess_sum').values())
        # Get mean excess
        excess_mean = stat.mean(nx.get_node_attributes(subg,'excess_mean').values())
        # Get max severity and max excess
        if config['methods_parameters']['tail'] == 'right':
            severity_max = max(nx.get_node_attributes(subg,'severity_max').values())
            excess_max = max(nx.get_node_attributes(subg,'excess_max').values())
        elif config['methods_parameters']['tail'] == 'left':
            severity_max = min(nx.get_node_attributes(subg,'severity_max').values())
            excess_max = min(nx.get_node_attributes(subg,'excess_max').values())
        date_start = datetime.strptime(date, "%Y-%m-%dT%H-%M-%S")
        date_end = date_start + timedelta(days=(duration-1))
        formatted_date_start = date_start.strftime("%Y%m%d")
        formatted_date_start_c = date_start.strftime("%Y-%m-%d")
        formatted_date_end = date_end.strftime("%Y%m%d")
        id_d = f"{formatted_date_start}-{formatted_date_end}_{id:03}"
        data.append((id_d, formatted_date_start_c, duration, countries, volume, area_mean, area_max, severity_mean, severity_max, excess_sum, excess_mean, excess_max))

    if config['data_structure']['UoM'] == "K":
        UoM = "°C"
    else:
        UoM = config['data_structure']['UoM']
    time_res = config['data_structure']['temporal_resolution']

    # create a dataframe from the collected data
    catalogue = pd.DataFrame(data, columns=['Id', 'Date', f'Duration ({time_res})', 'Countries',
                                            'Volume (km2)', 'Mean area (km2)',
                                            'Max area (km2)', f'Mean severity ({UoM})', f'Max severity ({UoM})', f'Total excess ({UoM})', f'Mean excess ({UoM})', f'Max excess ({UoM})'])

    return catalogue

def save_catalogue(catalogue, output_dir, update):

    if update:
        # read the existing catalogue
        catalogue_file = os.path.join(output_dir, 'catalogue.csv')
        if os.path.exists(catalogue_file):
            existing_catalogue = pd.read_csv(catalogue_file)
            # append the new catalogue to the existing one
            updated_catalogue = pd.concat([existing_catalogue, catalogue], ignore_index=True)
            updated_catalogue = updated_catalogue.sort_values(by='Date').reset_index(drop=True)
            updated_catalogue.to_csv(os.path.join(output_dir, 'catalogue.csv'), index=False, mode = 'w')
        else:
            logging.warning('The catalogue file does not exist. A new one will be created even the parameter ''update'' = True.')
            catalogue.to_csv(os.path.join(output_dir, 'catalogue.csv'), index=False, mode = 'w') 
    else:
        catalogue.to_csv(os.path.join(output_dir, 'catalogue.csv'), index=False, mode = 'w')
        
    logging.info('The catalogue of the events has been saved successfully as csv file.')
    print('The catalogue of the events has been saved successfully as csv file.')

def save_event_network(network, time, output_dir, update):
        
    # create the directory to store the events tracking
    if not os.path.exists(os.path.join(output_dir, 'events_tracking')):
        os.makedirs(os.path.join(output_dir, f'events_tracking'))

    # one csv for each event
    for subg in network:
        pos = nx.get_node_attributes(subg, "pos").values()
        event_data = {
            'centroid_lat': [i[0] for i in pos],
            'centroid_lon': [i[1] for i in pos],
            'date': time[[int(tt) for tt in (nx.get_node_attributes(subg, "time").values())]],
            'area': list(nx.get_node_attributes(subg, "area").values()),
            'severity_mean': list(nx.get_node_attributes(subg, "severity_mean").values()),
            'severity_max': list(nx.get_node_attributes(subg, "severity_max").values()),
            'excess_sum': list(nx.get_node_attributes(subg, "excess_sum").values()),
            'excess_mean': list(nx.get_node_attributes(subg, "excess_mean").values()),
            'excess_max': list(nx.get_node_attributes(subg, "excess_max").values()),
            'timestamp': list(nx.get_node_attributes(subg, "timestamp_from_start").values())
        }
        id = int(min(subg.nodes()))
        date_start = datetime.strptime(event_data['date'][0], "%Y-%m-%dT%H-%M-%S")
        date_end = datetime.strptime(event_data['date'][-1], "%Y-%m-%dT%H-%M-%S")
        formatted_date_start = date_start.strftime("%Y%m%d")
        formatted_date_end = date_end.strftime("%Y%m%d")
        id_d = f"{formatted_date_start}-{formatted_date_end}_{id:03}"
        pd.DataFrame(event_data).to_csv(os.path.join(output_dir, 'events_tracking', f'{id_d}.csv'),
                                                        mode = 'x' if update else 'w')
        
    logging.info('The tracks of each event have been saved successfully as csv files.')
    print('The tracks of each event have been saved successfully as csv files.')

def save_event_3D(mat3D, var, lat, lon, time, output_dir, config, update):

    var_name = config['data_structure']['variable_name']
    hazard_type = config['methods_parameters']['hazard_type']
    format_output = config['output_options']['format_output']
    UoM = config['data_structure']['UoM']
    if UoM == "K":
        UoM = "°C"
    long_name = config['output_options']['long_name']
    standard_name = config['output_options']['standard_name']

    # create the directory to store the 3D events
    if not os.path.exists(os.path.join(output_dir, 'events_3D')):
        os.makedirs(os.path.join(output_dir, 'events_3D'))

    # one zarr file for each event
    ids = np.unique(mat3D)
    ids = ids[~np.isnan(ids)]
    for i in ids:
        event3D = np.copy(var)
        event3D[mat3D != i] = np.nan
        # remove the time steps for which there are only NaN values
        nan_indices = np.isnan(event3D).all(axis=(1, 2))
        event3D = event3D[~nan_indices, :, :]
        time_event = time[~nan_indices]

        time_datetime = [datetime.strptime(t, "%Y-%m-%dT%H-%M-%S") for t in time_event]

        # Create Dataset
        event3D_ds = xr.Dataset(
            data_vars={
                var_name: (
                    ["time", "lat", "lon"], 
                    event3D,
                    {
                        "units": UoM,
                        "long_name": long_name,
                        "_FillValue": np.nan,
                        "standard_name": standard_name,
                    }
                )
            },
            coords={
                "time": (["time"], time_datetime, {"long_name": "time"}),
                "lat": (["lat"], lat, {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                }),
                "lon": (["lon"], lon, {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                }),
            },
            attrs={
                "description": f"{hazard_type} event",
                "Conventions": "CF-1.8",
                "history": f"Created on {datetime.now():%Y-%m-%d}",
                "source": "Generated by RHITA v1.0 (Real-time Hazards Identification and Tracking Algorithm)",
            }
        )

        # save the dataset for each event
        date_start = datetime.strptime(time_event[0], "%Y-%m-%dT%H-%M-%S")
        date_end = datetime.strptime(time_event[-1], "%Y-%m-%dT%H-%M-%S")
        formatted_date_start = date_start.strftime("%Y%m%d")
        formatted_date_end = date_end.strftime("%Y%m%d")
        id_d = f"{formatted_date_start}-{formatted_date_end}_{int(i):03}"
        if format_output == 'netcdf':
            if os.path.exists(os.path.join(output_dir, 'events_3D', f'{id_d}.nc4')) and update:
                logging.error('The file already exists. Please set the parameter ''update'' to False to overwrite the file.')
            else:
                event3D_ds.to_netcdf(os.path.join(output_dir, 'events_3D', f'{id_d}.nc4'), mode='w')
        elif format_output == 'zarr':
            if os.path.exists(os.path.join(output_dir, 'events_3D', f'{id_d}.zarr')) and update:
                logging.error('The file already exists. Please set the parameter ''update'' to False to overwrite the file.')
            else:
                event3D_ds.to_zarr(os.path.join(output_dir, 'events_3D', f'{id_d}.zarr'), mode='w')
        else:
            logging.error('The format for save the detected events must be "netcdf" or "zarr" but received "{}"'.format(format_output))
            raise ValueError('The format for save the detected events must be "netcdf" or "zarr" but received "{}"'.format(format_output))

# Function for saving the output files
@utils.log_execution_time
def save_files(network, mat3D, var, lat, lon, time, config):
    
    hazard_type = config['methods_parameters']['hazard_type']
    output_folder = config['directories']['output']

    # create output directory if it does not exist
    output_dir = os.path.join(output_folder, hazard_type, config['output_options']['folder_name_hazard'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    update = config['output_options'].getboolean('update')

    if config['output_options'].getboolean('catalogue'):
        catalogue = create_catalogue(network, time, config)
        save_catalogue(catalogue, output_dir, update)

    if config['output_options'].getboolean('single_event_tracking'):
        save_event_network(network, time, output_dir, update)

    if mat3D is not None:
        save_event_3D(mat3D, var, lat, lon, time, output_dir, config, update)