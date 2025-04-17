import numpy as np

def interpolate_rgb_colors(a, b, n):
    # interpolate a total of n colors between colors a and b (included in the total)
    res = np.zeros([n, 3])
    for i in range(3):
        res[0, i] = a[i]
        res[-1, i] = b[i]
    for j in range(1, n - 1):
        res[j, :] = ((n - 1 - j) / (n - 1)) * res[0, :] + (j / (n - 1)) * res[-1, :]
    return [tuple(res[k, :]) for k in range(n)]

def discrete_to_continous_cmap(list_colors, N=256):
    # number of base colors from which to interpolate
    ncolors = len(list_colors)

    # number of colors per bin (i.e. between to base colors)
    nc_per_bin = N // (ncolors - 1)

    # number of bins to process
    remain = N % (ncolors - 1)

    # resulting colormap
    res = []

    # loop on the bins
    for ibin in range(ncolors - 1):
        nbintemp = nc_per_bin
        if remain > 0:
            nbintemp = nbintemp + 1
            remain = remain - 1
        if ibin == ncolors - 2:
            #             print(len(interpolate_rgb_colors(list_colors[ibin],list_colors[ibin+1],nbintemp)[0:]))
            res = (
                res
                + interpolate_rgb_colors(
                    list_colors[ibin], list_colors[ibin + 1], nbintemp
                )[0:]
            )
        else:
            #             print(len(interpolate_rgb_colors(list_colors[ibin],list_colors[ibin+1],nbintemp+1)[:-1]))
            res = (
                res
                + interpolate_rgb_colors(
                    list_colors[ibin], list_colors[ibin + 1], nbintemp + 1
                )[:-1]
            )
    return res

def cmap_heavy_prec(n_colors):
    list_colors = [
       # (153 / 255, 210 / 255, 245 / 255),
       # (40 / 255, 115 / 255, 248 / 255),
       # (130 / 255, 92 / 255, 230 / 255),
       # (171 / 255, 86 / 255, 221 / 255),
       # (229 / 255, 181 / 255, 95 / 255),
       # (250 / 255, 252 / 255, 31 / 255)


        (0, 0, 0.5),   # Dark Blue
        (0, 0, 1),     # Blue
        (0, 0.7, 1),   # Cyan-Blue
        (0, 1, 1),     # Cyan
        (0.5, 1, 0.5), # Light Green
        (1, 1, 0),     # Yellow
        (1, 0.7, 0),   # Orange-Yellow
        (1, 0.4, 0),   # Orange
        (1, 0, 0),     # Red
        (0.5, 0, 0)    # Dark Red

     #   (0.        , 0.        , 0.5),
     #   (0.        , 0.3       , 1),
     #   (0.16129032, 1.        , 0.80645161),
     #   (0.80645161, 1.        , 0.16129032),
     #   (1.        , 0.40740741, 0),
     #   (0.5       , 0.        , 0)
    ]

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res

def cmap_heavy_prec_drama(n_colors = 15):
    list_colors = [
        (2 / 255, 144 / 255, 214 / 255),
        (2 / 255, 116 / 255, 173 / 255),
        (1 / 255, 78 / 255, 140 / 255),
        (1 / 255, 62 / 255, 112 / 255),
        (1 / 255, 38 / 255, 69 / 255),
        (0 / 255, 27 / 255, 48 / 255)
    ]

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res

def cmap_heatwave(n_colors):
    list_colors = [
        (247/255, 247/255, 247/255),
        (253/255, 219/255, 199/255),
        (244/255, 165/255, 130/255),
        (214/255, 96/255, 77/255),
        (178/255, 24/255, 43/255),
        (103/255, 0/255, 31/255)]

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res

def cmap_cold_spell(n_colors):
    list_colors = [
        (5/255, 48/255, 97/255),
        (33/255, 102/255, 172/255),
        (67/255, 147/255, 195/255),
        (146/255, 197/255, 222/255),
        (209/255, 229/255, 240/255),
        (247/255, 247/255, 247/255)]        

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res

def cmap_drought(n_colors):
    list_colors = [
        (128/255, 0/255, 0/255),      # Deep Red (Highest Alert)
        (178/255, 34/255, 34/255),    # Dark Red
        (255/255, 0/255, 0/255),      # Red
        (230/255, 97/255, 0/255),     # Dark Orange
        (255/255, 140/255, 0/255),    # Orange
        (255/255, 179/255, 71/255),   # Light Orange
        (255/255, 215/255, 0/255),    # Yellow
        (255/255, 255/255, 153/255)   # Light Yellow (Lowest Alert)
    ]

#        (84/255, 48/255, 5/255),
#                   (140/255, 81/255, 10/255),
#                   (191/255, 129/255, 45/255),
#                   (223/255, 194/255, 125/255),
#                   (246/255, 232/255, 195/255),
#                   (245/255, 245/255, 245/255)]

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res  

def cmap_wind(n_colors):
    list_colors = [
        (255/255, 255/255, 128/255),
        (255/255, 255/255, 0/255),
        (255/255, 128/255, 0/255),
        (251/255, 26/255, 26/255),
        (128/255, 0/255, 0/255)]

    res = discrete_to_continous_cmap(list_colors, N=n_colors)
    return res  

def list_colors(hazard, n_colors):
    if hazard == 'heavy_precipitation':
        list = cmap_heavy_prec(n_colors)
        # colormap for figure of the movie
        #cmap = plt.cm.jet  # define the colormap
        ## extract all colors from the .jet map
        #list = [cmap(i) for i in range(cmap.N)]
        ## create the new map
        #cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        # Create a ListedColormap from the discrete colormap
        #list = ListedColormap(list(np.linspace(0, 1, 15)))
    elif hazard == 'heatwave':
        list = cmap_heatwave(n_colors)
    elif hazard == 'cold_spell':
        list = cmap_cold_spell(n_colors)
    elif hazard == 'drought':
        list = cmap_drought(n_colors)
    elif hazard == 'wind':
        list = cmap_wind(n_colors)

    return list