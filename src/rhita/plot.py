import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import logging
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import xarray as xr
import imageio.v2 as iio
import rhita.create_cmap as cmap
from rhita.utils import log_execution_time

# Function for plotting the hazard's catalogue
@log_execution_time
def plot_catalogue(config, n_colors = 10):

    # load the hazard catalogue as a dataframe
    catalogue = pd.read_csv(os.path.join(config['directories']['input'], 'catalogue.csv'), header=0, sep=',')
    catalogue['Date'] = pd.to_datetime(catalogue['Date'], format='%Y-%m-%dT%H-%M-%S')
    if len(catalogue) == 1:
        logging.warning("The historical catalogue contains only one event, the catalogue plot has not been created")
        warnings.warn("The catalogue plot has not been created: the historical catalogue contains only one event")
        return

    # extract the configuration parameters
    UoM = config['data_structure']['UoM']
    t_res = config['data_structure']['temporal_resolution']
    variable_name = config['data_structure']['variable_name']
    dir_fig = os.path.join(config['directories']['plots'], config['data_structure']['hazard'], config['figures']['figure_folder_name'])
    format_type = config['settings']['format']
    if format_type == 'png' or format_type == 'jpg' or format_type == 'jpeg': 
        dpi = int(config['settings']['fig_res'])

    # set plotting parameters
    max_size = 1500
    min_size = 100
    mean_size = (max_size + min_size) / 2
    legend_sizes = [min_size, mean_size, max_size]
    max_vol = max(catalogue['Volume (km2)'])
    min_vol = min(catalogue['Volume (km2)'])
    mean_vol = (max_vol + min_vol) / 2  
    legend_vols = [min_vol, mean_vol, max_vol]          
    dot_sizes = (catalogue['Volume (km2)'] - min_vol) * (max_size - min_size) / (max_vol - min_vol) + min_size
    if config['data_structure']['hazard'] == 'heavy_precipitation':
        #lowest_range = int(np.floor(min(catalogue[f'Mean severity ({UoM})'])) / 5) * 5
        lowest_range = int(np.floor(min(catalogue[f'Max severity ({UoM})'])) / 5)
        #highest_range = int(np.ceil(max(catalogue[f'Mean severity ({UoM})'])) / 5) * 5
        highest_range = int(np.ceil(max(catalogue[f'Max severity ({UoM})'])) / 5)
        highest_range = int(np.ceil(max(catalogue[f'Max severity ({UoM})'])) / 5) + 5
    elif config['data_structure']['hazard'] == 'drought':
        lowest_range = int(np.floor(min(catalogue[f'Mean severity ({UoM})'])))
        highest_range = int(np.ceil(max(catalogue[f'Mean severity ({UoM})'])))
    elif config['data_structure']['hazard'] == 'cold_spell' or config['data_structure']['hazard'] == 'heatwave':
        lowest_range = int(np.floor(min(catalogue[f'Max severity ({UoM})'])) / 2) * 2
        highest_range = int(np.ceil(max(catalogue[f'Max severity ({UoM})'])) / 2) * 2
    elif config['data_structure']['hazard'] == 'wind':
        lowest_range = int(np.floor(min(catalogue[f'Mean severity ({UoM})'])))
        highest_range = int(np.ceil(max(catalogue[f'Mean severity ({UoM})'])))

    # create a discrete colormap
    discrete_cmap = mcolors.ListedColormap(cmap.list_colors(config['data_structure']['hazard'], n_colors))
    
    # manage duration and its unit of measure
    if t_res == '30 min':
        duration = catalogue[f'Duration ({t_res})'] * 30 / 60
        t_UoM = 'h'
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '5 min':
        duration = catalogue[f'Duration ({t_res})'] * 5 / 60
        t_UoM = 'h'
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '1 hour':
        duration = catalogue[f'Duration ({t_res})'] 
        t_UoM = 'h'
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '6 hours':
        duration = catalogue[f'Duration ({t_res})'] * 6
        t_UoM = 'h'
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '3 hours':
        duration = catalogue[f'Duration ({t_res})'] * 3
        t_UoM = 'h'  
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '1 day':
        duration = catalogue[f'Duration ({t_res})']
        t_UoM = 'day'
        time = catalogue['Date'].dt.strftime('%d %b %Y')
    elif t_res == '1 month':
        duration = catalogue[f'Duration ({t_res})']
        t_UoM = 'month'
        time = catalogue['Date'].dt.strftime('%b %Y')
    elif t_res == '1 year':
        duration = catalogue[f'Duration ({t_res})']
        t_UoM = 'year'
        time = catalogue['Date'].dt.strftime('%Y')

    # set the font size
    plt.rcParams.update({'font.size': 24})  # Change the default font size
    plt.rcParams['xtick.labelsize'] = 20  # Font size for x-axis ticks
    plt.rcParams['ytick.labelsize'] = 20  # Font size for y-axis ticks
    plt.rcParams['axes.labelsize'] = 24  # Font size for axis labels
    plt.rcParams['axes.titlesize'] = 24  # Font size for plot title 
    plt.rcParams['legend.fontsize'] = 22  # Font size for legend
    plt.rcParams['legend.title_fontsize'] = 24  # Font size for legend title

    # create the plot   
    fig, ax = plt.subplots(figsize=(25, 10))
    sc = ax.scatter(catalogue['Date'], duration, s=dot_sizes,
                    c=catalogue[f'Max severity ({UoM})'], cmap=discrete_cmap,
                    vmin=lowest_range, vmax=highest_range, alpha=0.7, edgecolors='black')
    plt.colorbar(sc, ax=ax, 
                        ticks=np.linspace(lowest_range, highest_range, (n_colors + 1)),
                        label = f'Max {variable_name} ({UoM})')
    for i, vv in enumerate(legend_vols):
        ax.scatter([], [], s=legend_sizes[i], label=str(round(vv)) + ' km²', color='gray', alpha=0.7)
    legend = ax.legend(loc = "upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(legend_sizes), frameon = False)

    legend.set_title('Total impacted area (km²)', prop={'size': 18})
    plt.xlabel('Date')
    plt.ylabel(f'Duration ({t_UoM})')
    plt.xticks(rotation=45)
    if t_res == '30 min' or t_res == '5 min' or t_res == '6 hours' or t_res == '3 hours':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    elif t_res == '1 month':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    elif t_res == '1 year':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # select the threshold for the plotting of the labels (most extreme events)
    th_volume = np.percentile(catalogue[f'Volume (km2)'], 95)
    th_duration = np.percentile(duration, 95)
   # if (config['data_structure']['tail'] == 'right'):
   #     th_intensity = np.percentile(catalogue[f'Mean severity ({UoM})'], 95)
   #     for i, txt in enumerate(time):
   #         if(duration[i] > th_duration or catalogue[f'Mean severity ({UoM})'].iloc[i] > th_intensity or catalogue['Volume (km2)'].iloc[i] > th_volume):
   #             ax.annotate(txt, (catalogue['Date'].iloc[i], duration[i]), rotation = 0)
   # elif (config['data_structure']['tail'] == 'left'):
   #     th_intensity = np.percentile(catalogue[f'Mean severity ({UoM})'], 5)
   #     for i, txt in enumerate(time):
   #         if(duration[i] > th_duration or catalogue[f'Mean severity ({UoM})'].iloc[i] < th_intensity or catalogue['Volume (km2)'].iloc[i] > th_volume):
   #             ax.annotate(txt, (catalogue['Date'].iloc[i], duration[i]), rotation = 0)  

    plt.grid()
    #plt.tight_layout()
    
    # save the plot
    if format_type == 'png' or format_type == 'jpg' or format_type == 'jpeg':
        fig.savefig(os.path.join(dir_fig, f'catalogue.{format_type}'), dpi=dpi)
    else:
        fig.savefig(os.path.join(dir_fig, f'catalogue.{format_type}'))
    plt.close(fig)
    logging.info("Plot of historical hazards created successfully")    
    print("Plot of historical hazards created successfully")

@log_execution_time
def track_event(config, n_colors):

    # extract the configuration parameters
    UoM = config['data_structure']['UoM']
    var_name = config['data_structure']['variable_name']
    fig_var_name = config['figures']['figure_variable_name']
    format_type = config['settings']['format']
    if format_type == 'png' or format_type == 'jpg' or format_type == 'jpeg': 
        dpi = int(config['settings']['fig_res'])

    # directories
    dir_input = os.path.join(config['directories']['input'], 'events_tracking')
    dir_fig = os.path.join(config['directories']['plots'], config['data_structure']['hazard'], config['figures']['figure_folder_name'])
    # upload the single event time series
    if config['figures']['select_events'] == 'ID':
        eventID = config['figures']['ids']
        IDs = [int(x.strip()) for x in eventID.split(',')]
        files = [filename for ID in IDs
            for filename in os.listdir(dir_input)
            if filename.startswith(f'event{ID}_') and filename.endswith('.csv')
            ]   
    elif config['figures']['select_events'] == 'date':
        eventStartDate = config['figures']['starting_dates']
        eventEndDate = config['figures']['ending_dates']
        dateStart_list = [x.strip() for x in eventStartDate.split(',')]
        dateEnd_list = [x.strip() for x in eventEndDate.split(',')]
        files = [filename for start, end in zip(dateStart_list, dateEnd_list)
            for filename in os.listdir(dir_input)
            if start in filename and end in filename and filename.endswith('.csv')]
        IDs = [int(ff.split('event')[1].split('_')[0]) for ff in files]
    elif config['figures']['select_events'] == 'all':
        catalogue = pd.read_csv(os.path.join(config['directories']['input'], 'catalogue.csv'), header=0, sep=',')
        IDs = catalogue['Id'].values
        files = [filename for ID in IDs
            for filename in os.listdir(dir_input)
            if filename.startswith(f'event{ID}_') and filename.endswith('.csv')
            ] 

    plt.rcParams.update({'font.size': 22})  # Change the default font size
    plt.rcParams['xtick.labelsize'] = 16  # Font size for x-axis ticks
    plt.rcParams['ytick.labelsize'] = 16  # Font size for y-axis ticks
    plt.rcParams['axes.labelsize'] = 22  # Font size for axis labels
    plt.rcParams['axes.titlesize'] = 22  # Font size for plot title 
    plt.rcParams['legend.fontsize'] = 20  # Font size for legend
    plt.rcParams['legend.title_fontsize'] = 22  # Font size for legend title

    # loop over the files
    for ff, id in zip(files, IDs):

        event = pd.read_csv(os.path.join(config['directories']['input'], 'events_tracking', ff), header=0, sep=',', index_col=0)
        # sort dataframe by timestamp
        event = event.sort_values(by='timestamp') 

        # create two discrete colormaps (for event severity and timestamp)
        discrete_cmap = mcolors.ListedColormap(cmap.list_colors(config['data_structure']['hazard'], n_colors))
        if config['data_structure']['hazard'] == 'heavy_precipitation':
            lowest_range = int(np.floor(min(event[f'severity_mean'])) / 5) * 5
            highest_range = int(np.ceil(max(event[f'severity_mean'])) / 5) * 5
        elif config['data_structure']['hazard'] == 'drought':
            lowest_range = -3 #int(np.floor(min(event[f'severity_mean'])))
            highest_range = int(np.ceil(max(event[f'severity_mean'])))
        elif config['data_structure']['hazard'] == 'cold_spell' or config['data_structure']['hazard'] == 'heatwave':
            lowest_range = int(np.floor(min(event[f'severity_mean'])) / 2) * 2
            highest_range = int(np.ceil(max(event[f'severity_mean'])) / 2) * 2
        elif config['data_structure']['hazard'] == 'wind':
            lowest_range = int(np.floor(min(event[f'severity_mean'])))
            highest_range = int(np.ceil(max(event[f'severity_mean'])))

        n_times = max(event['timestamp']) + (2 if max(event['timestamp']) == 0 else 1)
        cmap_ff = plt.get_cmap('Greys')
        colors_edge = [cmap_ff(i / (n_times - 1)) for i in event['timestamp']]
        colors_edge_unique = [cmap_ff(i / (n_times - 1)) for i in np.unique(event['timestamp'])]
        edge_cmap = mcolors.ListedColormap(colors_edge_unique)
        edge_lowest_range = min(event['timestamp']) - 1
        edge_highest_range = max(event['timestamp']) 

        if config['subsetting']['spatial_subset_method'] == 'mask':
            mask_file = os.path.join(config['directories']['mask'], config['subsetting']['mask_file'])
            sf = gpd.read_file(mask_file)
            lon_min, lat_min, lon_max, lat_max = sf.geometry.total_bounds
        elif config['subsetting']['spatial_subset_method'] == 'coordinates':
            lon_min = float(config['subsetting']['lon_min'])
            lat_min = float(config['subsetting']['lat_min'])
            lon_max = float(config['subsetting']['lon_max'])
            lat_max = float(config['subsetting']['lat_max'])
        elif config['subsetting']['spatial_subset_method'] == 'event': 
            lon_min = event['centroid_lon'].min() - 2
            lat_min = event['centroid_lat'].min() - 2
            lon_max = event['centroid_lon'].max() + 2
            lat_max = event['centroid_lat'].max() + 2

        # create the plot
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        for i in event['timestamp']: 
            idx_t1 = event[event['timestamp'] == i].index
            idx_t2 = event[event['timestamp'] == (i + 1)].index
            for t1 in idx_t1:
                for t2 in idx_t2:
                    ax.plot(event['centroid_lon'][[t1,t2]], event['centroid_lat'][[t1,t2]], color='gray',
                    linewidth = 1, transform=ccrs.PlateCarree())
                    if config['figures']['plot_arrows'] == 'True':
                        mid_lon = (event['centroid_lon'][t1] + event['centroid_lon'][t2]) / 2
                        mid_lat = (event['centroid_lat'][t1] + event['centroid_lat'][t2]) / 2
                        ax.annotate('', xy=(mid_lon, mid_lat),
                            xytext=(event['centroid_lon'][t1], event['centroid_lat'][t1]),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                            transform=ccrs.PlateCarree())
        sc = ax.scatter(event['centroid_lon'], event['centroid_lat'], c=event['severity_mean'], 
               cmap=discrete_cmap, vmin=lowest_range, vmax=highest_range,
               transform=ccrs.PlateCarree(),
               s=200, alpha = 1, edgecolors=colors_edge, linewidth=2)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        fig.subplots_adjust(left = 0.1, right=0.7, top=0.9, bottom=0.1)
        edge_sc = ax.scatter([], [], c=[], cmap=edge_cmap, vmin=edge_lowest_range, vmax=edge_highest_range)
        cbar1_ax = fig.add_axes([0.72, 0.1, 0.02, 0.8])
        cbar1 = plt.colorbar(sc, ax = ax, cax = cbar1_ax,
                ticks=np.linspace(lowest_range, highest_range, (n_colors + 1)),
                label = f'Mean {fig_var_name} ({UoM})', shrink = 0.7)
        cbar2_ax = fig.add_axes([0.82, 0.1, 0.02, 0.8])
        cbar2 = plt.colorbar(edge_sc, ax = ax, cax = cbar2_ax,
                         ticks = np.unique(event['timestamp'])[::5] - 0.5,
                         label = f'Timestamp', shrink = 0.7)
        cbar2.ax.yaxis.set_label_position('right')
        cbar2.ax.yaxis.set_ticks_position('right')
        cbar2.ax.set_yticklabels(np.unique(event['date'])[::5]) 
        # add map features
        ax.add_feature(cfeat.LAND)
        ax.add_feature(cfeat.OCEAN)
        ax.add_feature(cfeat.COASTLINE)
        ax.add_feature(cfeat.BORDERS)
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # save the plot
        if format_type == 'png' or format_type == 'jpg' or format_type == 'jpeg':
            fig.savefig(os.path.join(dir_fig, f'event{id}_track.{format_type}'), dpi=dpi)
        else:
            fig.savefig(os.path.join(dir_fig, f'event{id}_track.{format_type}'))
        plt.close(fig)
    #if len(IDs) > 1:
    logging.info("Plot/s of spatiotemporal tracks created successfully")    
    print("Plot/s of spatiotemporal tracks created successfully")
    #else:
    #   logging.info("Plot of events spatio-temporal track created successfully") 
        
@log_execution_time
def movie(config, n_colors):

    UoM = config['data_structure']['UoM']
    var_name = config['data_structure']['variable_name']
    var_name_fig = config['figures']['figure_variable_name']

    # directories
    dir_input = os.path.join(config['directories']['input'], 'events_3D')
    dir_fig = os.path.join(config['directories']['plots'], config['data_structure']['hazard'], config['figures']['figure_folder_name'])
    
    # upload the single event time series
    if config['figures']['select_events'] == 'ID':
        eventID = config['figures']['ids']
        IDs = [int(x.strip()) for x in eventID.split(',')]
        files = [filename for ID in IDs
            for filename in os.listdir(dir_input)
            if filename.startswith(f'event{ID}_') and filename.endswith('.zarr')]   
    elif config['figures']['select_events'] == 'date':
        eventStartDate = config['figures']['starting_dates']
        eventEndDate = config['figures']['ending_dates']
        dateStart_list = [x.strip() for x in eventStartDate.split(',')]
        dateEnd_list = [x.strip() for x in eventEndDate.split(',')]
        files = [filename for start, end in zip(dateStart_list, dateEnd_list)
            for filename in os.listdir(dir_input)
            if start in filename and end in filename and filename.endswith('.zarr')]
        IDs = [int(ff.split('event')[1].split('_')[0]) for ff in files]
    elif config['figures']['select_events'] == 'all':
        catalogue = pd.read_csv(os.path.join(config['directories']['input'], 'catalogue.csv'), header=0, sep=',')
        IDs = catalogue['Id'].values
        files = [filename for ID in IDs
            for filename in os.listdir(dir_input)
            if filename.startswith(f'event{ID}_') and filename.endswith('.zarr')]   

    plt.rcParams.update({'font.size': 22})  # Change the default font size
    plt.rcParams['xtick.labelsize'] = 16  # Font size for x-axis ticks
    plt.rcParams['ytick.labelsize'] = 16  # Font size for y-axis ticks
    plt.rcParams['axes.labelsize'] = 22  # Font size for axis labels
    plt.rcParams['axes.titlesize'] = 22  # Font size for plot title 
    plt.rcParams['legend.fontsize'] = 20  # Font size for legend
    plt.rcParams['legend.title_fontsize'] = 22  # Font size for legend title

    for ff, id in zip(files, IDs):

        # create folder 
        movie_dir = os.path.join(dir_fig, ff.split('.')[0])
        # check if the folder exitst, if not create it
        if not os.path.exists(movie_dir):
            os.makedirs(movie_dir)
        # load the zarr file
        ds = xr.open_zarr(os.path.join(dir_input, ff))
        var = ds[var_name].values
        lon = ds['lon'].values
        lat = ds['lat'].values
        time = ds['time'].values
        
        discrete_cmap = mcolors.ListedColormap(cmap.list_colors(config['data_structure']['hazard'], n_colors))
        if config['data_structure']['hazard'] == 'heavy_precipitation':
            lowest_range = int(np.floor(np.nanmin(var)) / 5) * 5
            highest_range = int(np.ceil(np.nanmax(var)) / 5) * 5
        elif config['data_structure']['hazard'] == 'drought':
            lowest_range = -3 #int(np.floor(np.nanmin(var)))
            highest_range = int(np.ceil(np.nanmax(var)))
        elif config['data_structure']['hazard'] == 'cold_spell' or config['data_structure']['hazard'] == 'heatwave':
            lowest_range = int(np.floor(np.nanmin(var)) / 2) * 2
            highest_range = int(np.ceil(np.nanmax(var)) / 2) * 2
        elif config['data_structure']['hazard'] == 'wind':
            lowest_range = int(np.floor(np.nanmin(var)))
            highest_range = int(np.ceil(np.nanmax(var)))

        time_lab = [t.replace('2014', '2029') for t in time]
        for t in range(len(time)):
            
            fig, ax = plt.subplots(figsize = (10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            # plot the initial frame
            im = ax.pcolormesh(lon, lat, var[t, :, :], transform=ccrs.PlateCarree(), cmap = discrete_cmap,
                       vmin = lowest_range, vmax = highest_range)
            #var[t, :, :] = var[t, :, :]*2.5
            #var[var >= 90] = 89
            #im = ax.contourf(lon, lat, var[t, :, :], transform=ccrs.PlateCarree(), cmap = discrete_cmap,
            #           vmin = lowest_range, vmax = highest_range)
            if config['subsetting']['spatial_subset_method'] == 'mask':
                mask_file = os.path.join(config['directories']['mask'], config['subsetting']['mask_file'])
                sf = gpd.read_file(mask_file)
                lon_min, lat_min, lon_max, lat_max = sf.geometry.total_bounds
                ax.set_xlim([lon_min, lon_max])
                ax.set_ylim([lat_min, lat_max])
            elif config['subsetting']['spatial_subset_method'] == 'coordinates':
                lon_min = float(config['subsetting']['lon_min'])
                lat_min = float(config['subsetting']['lat_min'])
                lon_max = float(config['subsetting']['lon_max'])
                lat_max = float(config['subsetting']['lat_max'])
                ax.set_xlim([lon_min, lon_max])
                ax.set_ylim([lat_min, lat_max])
            else:
                ax.set_xlim([lon.min(), lon.max()])
                ax.set_ylim([lat.min(), lat.max()])
            ax.add_feature(cfeat.LAND)
            ax.add_feature(cfeat.OCEAN)
            ax.add_feature(cfeat.COASTLINE)
            ax.add_feature(cfeat.BORDERS, linestyle=':')
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
            plt.title(f'{time_lab[t]}')
            plt.tight_layout()
            fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = 0.8,
                        ticks = np.linspace(lowest_range, highest_range, (n_colors + 1)),
                        label = f'{var_name_fig} ({UoM})')
            #cbar = fig.colorbar(im, ax = ax, orientation = 'vertical', shrink = 0.6, label = 'Pluie horaire (mm/h)')              
            #ticks = np.linspace(0, 90, 19)
            #cbar.set_ticks(ticks[::2])
            #cbar.set_ticklabels([f'{int(tick)}' for tick in ticks[::2]])          
            fig.savefig(os.path.join(movie_dir, f't{'0' + str(t) if t < 10 else t}.png'), dpi=300)
            plt.close(fig)
        # get all the image files in the movie directory
        image_files = [f for f in os.listdir(movie_dir) if f.endswith('.png')]
        # sort the image files in ascending order
        image_files.sort()
        # create a list to store the images
        images = []
        # read each image file and append it to the list
        for file in image_files:
            image_path = os.path.join(movie_dir, file)
            image = iio.imread(image_path)
            images.append(image)

        # create the video from the images
        output_video = os.path.join(movie_dir, f'event{id}_movie.mp4')

        iio.mimsave(output_video, images, fps=2, macro_block_size=1)

    logging.info("Movie/s of spatiotemporal evolution created successfully")    
    print("Movie/s of spatiotemporal evolution created successfully")
