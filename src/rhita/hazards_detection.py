# This script contains functions for detecting hazards

import numpy as np
import logging
# import time
from rhita.utils import log_execution_time
from scipy.ndimage import label
import pandas as pd
import networkx as nx
from haversine import haversine_vector, Unit
import geopandas as gpd
from shapely.geometry import Point
from functools import partial
from multiprocessing import Pool
#import dask.array as da
#from dask.distributed import Client
import geopandas as gpd
from shapely.geometry import Point

# Function for binarizing the data based on a threshold 
@log_execution_time
def thresholding(var, var_th, time = None, time_th = None, config = None):

    # set the parameters
    tail = config['methods_parameters']['tail']
    method_th1 = config['methods_parameters']['threshold1']
    if method_th1 == "fixed":
        th1 = float(config['methods_parameters']['fixed_th1'])
    elif method_th1 == "quantile":
        th1 = np.quantile(var, float(config['methods_parameters']['level_quantile_th1']), axis = 0)
    elif method_th1 == "map":
        th1 = var_th
    elif method_th1 == 'map_time_of_year':
        th1 = var_th
    else:
        logging.error("Thresholding method not recognized")
        raise ValueError("Thresholding method not recognized")   
    
    if tail not in ["right", "left"]:
        logging.error("Tail must be 'right' or 'left'")
        raise ValueError("Tail must be 'right' or 'left'")
    
    # binarize the data
    if method_th1 == "map_time_of_year":
        # Create time of the year dataframe
        time_decoded = pd.to_datetime(time, format='%Y-%m-%dT%H-%M-%S')
        day_of_year = (time_decoded.dayofyear - 1).values  # Convert to 0-indexed        
        # Initialize an empty DataArray for binary maps
        binary_map = np.zeros_like(var, dtype=int)
        # Apply the day-specific threshold for each time step
        for t, doy in enumerate(day_of_year):
            binary_map[t, :, :] = var[t, :, :] >= var_th[doy, :, :]
    else:
        if tail == "right":
            binary_map = var >= th1 
        elif tail == "left":
            binary_map = var < th1
    
    return binary_map

# Functions for computing haversine distance
#def haversine_distance(point1, point2):
#    lat1, lon1 = np.radians(point1[0]), np.radians(point1[1])
#    lat2, lon2 = np.radians(point2[0]), np.radians(point2[1])
#    dlon = lon2 - lon1 
#    dlat = lat2 - lat1 
#    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#    # radius of Earth in kilometers is 6371
#    distance = 6371 * c
#    return distance

def degree_to_km(degree, lat):
    lat_rad = np.radians(lat)
    return degree * 111 * np.cos(np.radians(lat))

# Functions for identifying single events on the spatial domain
def identify_events(binary_map, lat_matrix, lon_matrix, distance_threshold):
 
    # find connected components in the matrix
    labeled_matrix, num_labels = label(binary_map)
    
    # dictionary to store event IDs and their pixel coordinates
    event_pixels = {}
    if num_labels != 0:
        # iterate over each labeled component
        for label_id in range(1, (num_labels + 1)):
            # find pixels belonging to the current component
            component_pixels = np.array(np.where(labeled_matrix == label_id)).T    
            # initialize a list to store IDs of connected events
            connected_event_ids = []    
            # iterate over existing events
            for event_id, pixels in event_pixels.items():
                # calculate the distance (in km) between every pixel of the current component and every pixel of the existing event
                sel_lat_1 = lat_matrix[pixels[:, 0], pixels[:, 1]]
                sel_lon_1 = lon_matrix[pixels[:, 0], pixels[:, 1]]
                sel_lat_2 = lat_matrix[component_pixels[:, 0], component_pixels[:, 1]]
                sel_lon_2 = lon_matrix[component_pixels[:, 0], component_pixels[:, 1]]
                # create meshgrid for all combinations
                lat1_grid, lat2_grid = np.meshgrid(sel_lat_1, sel_lat_2)
                lon1_grid, lon2_grid = np.meshgrid(sel_lon_1, sel_lon_2)
                # flatten the grids to create coordinate pairs
                lat1_flat = lat1_grid.flatten()
                lon1_flat = lon1_grid.flatten()
                lat2_flat = lat2_grid.flatten()
                lon2_flat = lon2_grid.flatten()
                points1 = np.column_stack((lat1_flat, lon1_flat))
                points2 = np.column_stack((lat2_flat, lon2_flat))
                # calculate distances between all combination of point using haversine_vector
                distances = haversine_vector(points1, points2, Unit.KILOMETERS)
                # find the minimum distance to any pixel of the existing event
                min_distance_to_event = np.min(distances)    
                # if the minimum distance is less than or equal to the threshold, add the event to the connected_event_ids list
                if min_distance_to_event <= distance_threshold:
                    connected_event_ids.append(event_id)
    
            # if the current component is close enough to at least two existing events, merge them all together
            if len(connected_event_ids) >= 1:
                combined_pixels = component_pixels
                #combined_ids = connected_event_ids.copy()
                for event_id in connected_event_ids:
                    combined_pixels = np.concatenate((combined_pixels, event_pixels[event_id]))
                    del event_pixels[event_id]
                event_pixels[label_id] = combined_pixels
            else:
                event_pixels[label_id] = component_pixels
    
            # create a new matrix with event IDs
            event_matrix = np.full(binary_map.shape, np.nan)
            for event_id, pixels in event_pixels.items():
                for pixel in pixels:
                    event_matrix[pixel[0], pixel[1]] = event_id
    else:
        event_matrix = np.full(binary_map.shape, np.nan)

    return event_matrix

# function to get the countries impacted by the event
def get_countries(lon, lat):  
    # Load country boundaries (Natural Earth or GADM)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Create GeoDataFrame of valid points
    points = gpd.GeoDataFrame(geometry=[Point(lon[i], lat[i]) for i in range(len(lon))], crs="EPSG:4326")
    # Spatial join to find countries
    countries = gpd.sjoin(points, world, how="left", predicate="within")
    # Get unique country names
    unique_countries = countries['name'].dropna().unique()
    return unique_countries

# functions for removing small events and redefine consecutive IDs
def filter_rename_2Devent(events, lat_mat, config):
    min_event_size = float(config['methods_parameters']['min_event_size'])
    i = 0
    events = np.array(events)
    event_ID2 = np.copy(events)
    for t in range(len(events)):
        temp = np.copy(events[t,:,:])
        IDs = np.unique(temp[~np.isnan(temp)])
        if len(IDs) > 0: 
            for aa in range(len(IDs)):
                cells_idx = np.where(temp == IDs[aa])
                dist_lon = degree_to_km(float(config['data_structure']['spatial_resolution']), lat_mat) # vertical size of the cells in km 
                dist_lat = float(config['data_structure']['spatial_resolution']) * 111 # horizontal size of the cells in km
                area_cell = dist_lon * dist_lat
                area_km = sum(area_cell[cells_idx])
                if area_km >= min_event_size:   
                    event_ID2[t,:,:][cells_idx] = int(i + 1)
                    i = i + 1 
                else: 
                    event_ID2[t,:,:][np.where(temp == IDs[aa])] = np.NaN
    return(event_ID2)   

@log_execution_time
def events_2D(binary_map, lat_mat, lon_mat, config):

    distance_th = float(config['methods_parameters']['distance_th_event2D'])

    if config['settings'].getboolean('parallel'):
        ncpu = config['settings'].getint('ncpu')
        # identify the events as 2D patches in parallel
        partial_identify_events = partial(identify_events, lat_matrix=lat_mat, lon_matrix=lon_mat, distance_threshold=distance_th)
        with Pool(ncpu) as pool:
            events = np.array(pool.map(partial_identify_events, binary_map))
    else:
        # identify the events as 2D patches
        events = np.array([identify_events(mm, lat_mat, lon_mat, distance_th) for mm in binary_map])
    events = filter_rename_2Devent(events, lat_mat, config)

    logging.info("Single events in the spatial domain identified successfully") 
    print("Single events in the spatial domain identified successfully")
    return events

@log_execution_time
def centroids(events, var, excess_var, lat_mat, lon_mat, config):

    method = config['methods_parameters']['method_centroid']
    if method not in ["weighted", "unweighted"]:
        logging.error(f"The variable 'method_centroid' must be 'weighted' or 'unweighted' but received '{method}'")
        raise ValueError(f"The variable 'method_centroid' must be 'weighted' or 'unweighted' but received '{method}'")
    tail = config['methods_parameters']['tail']
    if tail not in ["right", "left"]:
        logging.error(f"The variable 'tail' must be 'right' or 'left' but received '{tail}'")
        raise ValueError(f"The variable 'tail' must be 'right' or 'left' but received '{tail}'")

    # get unique IDs of events
    ID = np.unique(events[np.isfinite(events)])  # Changed to np.isfinite to handle NaN and Inf
    
    n_cells = np.zeros_like(ID, dtype=int)
    area_km = np.zeros_like(ID, dtype=float)
    severity_mean = np.zeros_like(ID, dtype=float)
    severity_max = np.zeros_like(ID, dtype=float)
    excess_sum = np.zeros_like(ID, dtype=float)
    excess_mean = np.zeros_like(ID, dtype=float)
    excess_max = np.zeros_like(ID, dtype=float)
    lat_mean = np.zeros_like(ID, dtype=float)
    lon_mean = np.zeros_like(ID, dtype=float)
    timestamp = np.zeros_like(ID, dtype=float)
    countries = np.empty_like(ID, dtype=object)
    
    # iterate over unique IDs and compute statistics
    for idx, i in enumerate(ID):
        cells_idx = np.where(events == i)
        n_cells[idx] = cells_idx[0].size
        dist_lon = degree_to_km(float(config['data_structure']['spatial_resolution']), lat_mat) # vertical size of the cells in km 
        dist_lat = float(config['data_structure']['spatial_resolution']) * 111 # horizontal size of the cells in km
        area_cell = dist_lon * dist_lat
        area_km[idx] = sum(area_cell[cells_idx[1:3]])
        severity_mean[idx] = np.mean(var[cells_idx])
        excess_sum[idx] = np.sum(excess_var[cells_idx])
        excess_mean[idx] = np.mean(excess_var[cells_idx])
        if(tail == "right"):
            severity_max[idx] = np.max(var[cells_idx])
            excess_max[idx] = np.max(excess_var[cells_idx])
        elif(tail == "left"):
            severity_max[idx] = np.min(var[cells_idx])
            excess_max[idx] = np.min(excess_var[cells_idx])
        if method == 'weighted' and n_cells[idx] > 1:
            lat_mean[idx] = np.average(lat_mat[cells_idx[1:3]], weights = abs(var[cells_idx]))
            lon_mean[idx] = np.average(lon_mat[cells_idx[1:3]], weights = abs(var[cells_idx]))
        else:
            lat_mean[idx] = np.mean(lat_mat[cells_idx[1:3]])
            lon_mean[idx] = np.mean(lon_mat[cells_idx[1:3]])
        if len(np.unique(cells_idx[0])) != 1:
            logging.error("Invalid Dimension: The computation of the centroid should be on 2D events, but 3D events were passed.")
            raise ValueError("Invalid Dimension: The computation of the centroid should be on 2D events, but 3D events were passed.")
        timestamp[idx] = np.unique(cells_idx[0])
        countries[idx] = get_countries(lon_mat[cells_idx[1:3]], lat_mat[cells_idx[1:3]])
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': ID,
        'lat': lat_mean,
        'lon': lon_mean,
        'area': area_km,
        'severity_mean': severity_mean,
        'severity_max': severity_max,
        'excess_sum': excess_sum,
        'excess_mean': excess_mean,
        'excess_max': excess_max,
        'timestamp': timestamp,
        'countries': countries
    })
    
    logging.info("Centroids for each spatial event computed successfully") 
    print("Centroids for each spatial event computed successfully")
    return df

def create_network(centroids, events, lat_mat, lon_mat, config):

    method = config['methods_parameters']['method_event3D']

    # create an empty directed graph
    G = nx.DiGraph()
    pos = {}
    timestamps = np.unique(centroids['timestamp']).astype(int)
    
    # create the nodes (which correspond to the centroids of the 2D events)
    for _, row in centroids.iterrows():
        node_attrs = {
            'pos': (row['lat'], row['lon']),
            'time': row['timestamp'],
            'area': row['area'], 
            'severity_mean': row['severity_mean'], 
            'severity_max': row['severity_max'],
            'excess_sum': row['excess_sum'], 
            'excess_mean': row['excess_mean'], 
            'excess_max': row['excess_max'], 
            'countries': row['countries']
        }
        id_t = row['ID']
        pos[id_t] = (row['lon'], row['lat'])
        G.add_node(id_t, **node_attrs)

    # connect centroids between consecutive timestamp with a specific method (selected among 'min_distance_cells', 'min_distance_centroids', 'overlapping_absolute_area', 'overlapping_relative_area')
    for t in timestamps:
        current_time_centroids = centroids[centroids['timestamp'] == t]
        next_time_centroids = centroids[centroids['timestamp'] == t + 1]     

        for nn, current_row in current_time_centroids.iterrows():
            for nn_next_time, next_row in next_time_centroids.iterrows():          
                current_centroid_pos = (current_row['lat'], current_row['lon'])
                next_centroid_pos = (next_row['lat'], next_row['lon'])              
                current_event_id = current_row['ID']
                next_event_id = next_row['ID']
                
                # Check if the minimum distance between events is above the threshold
                if method == 'min_distance_cells':
                    distance_threshold = float(config['methods_parameters']['distance_th_event3D'])

                    # calculate the distances among all the cells of the two events
                    pos_t0 = np.array(np.where(events[t, :, :] == current_row['ID'])).T
                    pos_t1 =  np.array(np.where(events[t + 1, :, :] == next_row['ID'])).T
                    sel_lat_1 = lat_mat[pos_t0[:, 0], pos_t0[:, 1]]
                    sel_lon_1 = lon_mat[pos_t0[:, 0], pos_t0[:, 1]]
                    sel_lat_2 = lat_mat[pos_t1[:, 0], pos_t1[:, 1]]
                    sel_lon_2 = lon_mat[pos_t1[:, 0], pos_t1[:, 1]]
                    # create meshgrid for all combinations
                    lat1_grid, lat2_grid = np.meshgrid(sel_lat_1, sel_lat_2)
                    lon1_grid, lon2_grid = np.meshgrid(sel_lon_1, sel_lon_2)
                    # flatten the grids to create coordinate pairs
                    lat1_flat = lat1_grid.flatten()
                    lon1_flat = lon1_grid.flatten()
                    lat2_flat = lat2_grid.flatten()
                    lon2_flat = lon2_grid.flatten()
                    points1 = np.column_stack((lat1_flat, lon1_flat))
                    points2 = np.column_stack((lat2_flat, lon2_flat))
                    # calculate distances between all combination of point using haversine_vector
                    distances = haversine_vector(points1, points2, Unit.KILOMETERS)
                    # find the minimum distance to any pixel of the existing event
                    min_distance_to_event = np.min(distances)
                    
                    # if the minimum distance is less than or equal to the threshold, add an edge between the two events' centroids
                    if min_distance_to_event <= distance_threshold:
                        G.add_edge(current_event_id, next_event_id)

                # Check if distance between centroids is less than or equal to the threshold 
                elif method == 'min_distance_centroids':
                    distance_threshold = float(config['methods_parameters']['distance_th_event3D'])
                    
                    # compute the distance between centroids
                    centroid_distance = haversine_vector(current_centroid_pos, next_centroid_pos)

                    # if the distance is less than or equal to the threshold, add an edge between the two events' centroids
                    if centroid_distance <= distance_threshold:
                        G.add_edge(current_event_id, next_event_id)

                # Check if the percentage of overlapping area between the two events is greater than threshold 
                elif method == 'overlapping_relative_area':
                    perc_area_threshold = float(config['methods_parameters']['perc_area_th']) 

                    # compute the percentage of overlapping area between event(t) and event(t+1) 
                    perc_overlapping1 = 100 * np.sum(np.logical_and(events[t, :, :] == current_event_id,
                                                                events[t + 1, :, :] == next_event_id)) / np.sum(events[t + 1, :, :] == next_event_id)
                    perc_overlapping2 = 100 * np.sum(np.logical_and(events[t, :, :] == current_event_id,
                                                                events[t + 1, :, :] == next_event_id)) / np.sum(events[t, :, :] == current_event_id)
                    # if there maximum between the two percentages is greater than or equal to the threshold, add an edge between the two events' centroids
                    if max(perc_overlapping1, perc_overlapping2) >= perc_area_threshold:
                        G.add_edge(current_event_id, next_event_id)

                elif method == 'overlapping_absolute_area':
                    cells_threshold = float(config['methods_parameters']['n_cells_th'])
                    n_cells_overlap = np.sum(np.logical_and(events[t, :, :] == current_row['ID'],
                                                            events[t + 1, :, :] == next_row['ID']))
                    
                    # if the number of overlapping cells is greater than or equal to the threshold, add an edge between the two events' centroids
                    if n_cells_overlap >= cells_threshold:
                        G.add_edge(current_event_id, next_event_id)

                else:
                    logging.error("The variable 'method_event3D' must be 'min_distance_cells', 'min_distance_centroids', 'overlapping_relative_area', or 'overlapping_absolute_area' but received '{}'".format(method))
                    raise ValueError("The variable 'method_event3D' must be 'min_distance_cells', 'min_distance_centroids', 'overlapping_relative_area', or 'overlapping_absolute_area' but received '{}'".format(method))

    # compute the shortest path lengths (duration from the beginning of the event) from the root node to each node
    longest_path_lengths = {}
    for root_node in G.nodes():
        # compute shortest path lengths from the root node using the reverse graph
        shortest_path_lengths = nx.single_source_shortest_path_length(G.reverse(), root_node)
        # find the longest path length among all paths
        longest_path_length = max(shortest_path_lengths.values())
        longest_path_lengths[root_node] = longest_path_length

    nx.set_node_attributes(G, longest_path_lengths, name='timestamp_from_start')

    return(G)

def impacted_area(mat, lat_mat, lon_mat, config):
    # compute the total impacted area
    dist_lon = degree_to_km(float(config['data_structure']['spatial_resolution']), lat_mat) # vertical size of the cells in km 
    dist_lat = float(config['data_structure']['spatial_resolution']) * 111 # horizontal size of the cells in km
    area_cell = dist_lon * dist_lat
    area_km = sum(area_cell[mat == 1])
    return area_km

@log_execution_time
def events_3D(centroids, events, lat_mat, lon_mat, config):
    
    # create the network for all the event
    network = create_network(centroids, events, lat_mat, lon_mat, config)
    # identify connected components into the network (G) and create a list of subgraphs (one network for each event)
    events_network = [network.subgraph(c).copy() for c in nx.connected_components(network.to_undirected())]
    # remove the subgraphs with duration (length) shorter than the threshold
    min_duration = float(config['methods_parameters']['min_duration']) - 1
    events_network = [subg for subg in events_network if max(nx.get_node_attributes(subg, "timestamp_from_start").values()) >= min_duration]
    # create a 3D matrix with the same shape of the original events matrix but rename with the IDs of the 3D event
    events_3D_id = np.copy(events)
    final_ID = np.array([])
    for ev in events_network:
        ID_all = ev.nodes
        ID_min = int(min(ID_all))
        events_3D_id[np.isin(events, ID_all)] = ID_min
        final_ID = np.append(final_ID, ID_min)
    # remove all the ID in events_3D_id that are not in final_ID
    events_3D_id[~np.isin(events_3D_id, final_ID)] = np.NaN
    # add a new attribute to the network with the total impacted area (the projection of the 3D event on the spatial domain)
    #for ev in events_network:
    #    min_node = min(ev.nodes)  # Get the minimum node ID
    #    impacted_mask = (events_3D_id == min_node)  # Create a 3D mask (time, lat, lon)
    #    # Collapse along time axis: Keep True at (lat, lon) if at least one time step is True
    #    impacted_mask_3D = impacted_mask.any(axis=0)  # Project along the time axis
    #    ev.graph['impacted_area'] = impacted_area(events_3D_id == min(ev.nodes), lat_mat, lon_mat, config)

    logging.info("3D events identified successfully")
    print("3D events identified successfully")
    return events_network, events_3D_id