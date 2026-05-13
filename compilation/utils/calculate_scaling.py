import xarray as xr

def calculate_sclaing_stats_input(file, channel_in):
    """
    Calculate the scaling statistics of channel_in input variables from file
    """

    ds = xr.open_dataset(file, engine='zarr')

    for c in channel_in: 

        da = ds.inputs.sel(channel_in=c)
        print('==============================================')
        print(f'Calculating Stats for {c}')
        da.load()
        print(f'  mean: {da.values.mean()}')
        print(f'  std: {da.values.std()}')
    
    print('done.')

def calculate_sclaing_stats_const(file, channel_c):
    """
    Calculate the scaling statistics of channel_c constant fields from file
    """

    ds = xr.open_dataset(file, engine='zarr')

    for c in channel_c: 

        da = ds.constants.sel(channel_c=c)
        print('==============================================')
        print(f'Calculating Stats for {c}')
        da.load()
        print(f'  mean: {da.values.mean()}')
        print(f'  std: {da.values.std()}')
    
    print('done.')

if __name__ == '__main__':

    # calculate_sclaing_stats_input('/home/disk/rhodium/dlwp/data/HPX64/hpx64_2000-2025_dltm-v4.zarr',
    #     channel_in = [
    #         'NDVI_gapfill', 'swvl1', 'swvl4', 'stl1', 'stl4', 'q2m-48H', 'tcwv-48H',
    #    'q850-48H', 'z1000-48H', 't2m-48H', 'ttr6-48H', 'tp6-48H',
    #     ])
    calculate_sclaing_stats_const('/home/disk/rhodium/dlwp/data/HPX64/hpx64_2000-2025_dltm-v4.zarr',
        channel_c = [
            'NDVI_gapfill_annual_range', 'NDVI_gapfill_min', 'NDVI_gapfill_max', 
            'swvl1_annual_range', 'swvl1_min', 'swvl1_max',
            'stl1_annual_range', 'stl1_min', 'stl1_max',
        ])
    