import pandas as pd
import scipy.stats as stats
import numpy as np

def fit_GEV(data, config):
    
    alpha = float(config['analyses']['alpha'])
    
    par_GEV = stats.genextreme.fit(data)
    # generate random numbers from fitted GEV distribution
    #n_events = 1000
    #random_numbers = stats.genextreme.rvs(par_GEV[0], par_GEV[1], par_GEV[2], size=n_events)
    # perform ad test
    #ad_test = stats.anderson_ksamp([data, random_numbers])
    # perform ks test
    ks_test = stats.kstest(data, 'genextreme', args=(par_GEV[0], par_GEV[1], par_GEV[2]))

    # plt empirical vs theoretical (GEV) CDF
    # plt.figure()
    # plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
    # plt.xlabel(f'Mean severity ({UoM})')
    # plt.ylabel('CDF')
    # x = np.linspace(min(data), max(data), 100)
    # y = stats.genextreme.cdf(x, par_GEV[0], par_GEV[1], par_GEV[2])
    # plt.plot(x, y, color='red')
    # plt.legend(['Empirical CDF', 'GEV CDF'])
    # plt.show()
    # plt.close()
    #significant_test = ad_test.pvalue > alpha and ks_test.pvalue > alpha
    #significant_test = ks_test.pvalue > alpha

    return ks_test.pvalue, par_GEV 

def fit_GPD(data, config):
    
    alpha = float(config['analyses']['alpha'])
    par_GPD = stats.genpareto.fit(data)
    ks_test = stats.kstest(data, 'genpareto', args=(par_GPD[0], par_GPD[1], par_GPD[2]))
    
    #significant_test = ks_test.pvalue > alpha

    return ks_test.pvalue, par_GPD 

def groupyear(year, nyears_grouped, first_year):

    return((year - first_year) // nyears_grouped) * nyears_grouped + first_year

def events_per_year(dates):
    
    years = pd.to_datetime(dates).dt.year
    # count the number of events per year
    events_per_year = years.value_counts()

    return events_per_year.mean(), events_per_year.std()

def trend_nevents(catalogue, config):

    nyears = float(config['analyses']['nyears'])
    datetime_array = pd.to_datetime(catalogue['Date'])
    catalogue['Year'] = datetime_array.dt.year

    if (np.unique(catalogue['Year']).size >= nyears*2):
        
        alpha = float(config['analyses']['alpha'])
    
        first_year = catalogue['Year'].min()
        catalogue[f'{nyears} ys'] = catalogue['Year'].apply(groupyear, args=(nyears, first_year,))
        last_rows = (catalogue.iloc[-1]['Year'] - first_year) % nyears
        if last_rows != 0:
            catalogue = catalogue.iloc[:-int(last_rows)]

        # trend analysis for severity
        nevents_per_group = catalogue.groupby(f'{nyears} ys').size()
        nevents_slope, _, _, nevents_p_value, _ = stats.linregress(np.linspace(0, len(nevents_per_group) - 1, len(nevents_per_group)), nevents_per_group.values)

        # plt.figure()
        # plt.plot(nevents_per_group.index, nevents_per_group.values, '-o')
        # plt.show()
        # plt.close()
        # severity_trend = mk.original_test(nevents_per_group.values)

        #significant_test = nevents_p_value < alpha

        return nevents_p_value, nevents_slope 

    else:

        print("Timeseries not long enough to perform the trend analysis on the number of events per year")
        return None, None

def trend_severity(catalogue, config):
    
    nyears = float(config['analyses']['nyears'])
    datetime_array = pd.to_datetime(catalogue['Date'])
    catalogue['Year'] = datetime_array.dt.year

    if np.unique(catalogue['Year']).size >= nyears*2:
        UoM = config['data_structure']['UoM']
        alpha = float(config['analyses']['alpha'])

        first_year = catalogue['Year'].min()
        catalogue[f'{nyears} ys'] = catalogue['Year'].apply(groupyear, args=(nyears, first_year,))
        last_rows = (catalogue.iloc[-1]['Year'] - first_year) % nyears
        if last_rows != 0:
            catalogue = catalogue.iloc[:-int(last_rows)]

        # trend analysis for severity
        severity_per_group = catalogue.groupby(f'{nyears} ys')[f'Mean severity ({UoM})'].mean()
        severity_slope, _, _, severity_p_value, _ = stats.linregress(np.linspace(0, len(severity_per_group) - 1, len(severity_per_group)), severity_per_group.values)

        # plt.figure()
        # plt.plot(severity_per_group.index, severity_per_group.values, '-o')
        # plt.show()
        # plt.close()
        # severity_trend = mk.original_test(severity_per_group.values)

        significant_test = severity_p_value < alpha

        return severity_p_value, severity_slope
    else:

        print("Timeseries not long enough to perform the trend analysis on the severity")
        return None, None
    
def trend_volume(catalogue, config):
    
    nyears = float(config['analyses']['nyears'])
    datetime_array = pd.to_datetime(catalogue['Date'])
    catalogue['Year'] = datetime_array.dt.year

    if (np.unique(catalogue['Year']).size >= nyears*2):

        alpha = float(config['analyses']['alpha'])

        first_year = catalogue['Year'].min()
        catalogue[f'{nyears} ys'] = catalogue['Year'].apply(groupyear, args=(nyears, first_year,))
        last_rows = (catalogue.iloc[-1]['Year'] - first_year) % nyears
        if last_rows != 0:
            catalogue = catalogue.iloc[:-int(last_rows)]

        # trend analysis for severity
        volume_per_group = catalogue.groupby(f'{nyears} ys')['Volume (km2)'].mean()
        volume_slope, _, _, volume_p_value, _ = stats.linregress(np.linspace(0, len(volume_per_group) - 1, len(volume_per_group)), volume_per_group.values)

        # plt.figure()
        # plt.plot(volume_per_group.index, volume_per_group.values, '-o')
        # plt.show()
        # plt.close()
        # severity_trend = mk.original_test(volume_per_group.values)

        significant_test = volume_p_value < alpha

        return volume_p_value, volume_slope 
    else:

        print("Timeseries not long enough to perform the trend analysis on the volume")
        return None, None

def trend_duration(catalogue, config):

    nyears = float(config['analyses']['nyears'])
    datetime_array = pd.to_datetime(catalogue['Date'])
    catalogue['Year'] = datetime_array.dt.year

    if (np.unique(catalogue['Year']).size >= nyears*2):
    
        t_res = config['data_structure']['temporal_resolution']
        alpha = float(config['analyses']['alpha'])

        first_year = catalogue['Year'].min()
        catalogue[f'{nyears} ys'] = catalogue['Year'].apply(groupyear, args=(nyears, first_year,))
        last_rows = (catalogue.iloc[-1]['Year'] - first_year) % nyears
        if last_rows != 0:
            catalogue = catalogue.iloc[:-int(last_rows)
                                       ]
        # trend analysis for severity
        duration_per_group = catalogue.groupby(f'{nyears} ys')[f'Duration ({t_res})'].mean()
        duration_slope, _, _, duration_p_value, _ = stats.linregress(np.linspace(0, len(duration_per_group) - 1, len(duration_per_group)), duration_per_group.values)

        significant_test = duration_p_value < alpha
        return duration_p_value, duration_slope

    else:

        print("Timeseries not long enough to perform the trend analysis on the duration")
        return None, None
    
def correlation(x, y):
    
    return stats.pearsonr(x, y)    