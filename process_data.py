import xarray as xr
import cfgrib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

# FourCastNet
# Read the grib-file
# - Here we use cfgrib.open_datasets because the grib message contains fields
#   on different types of levels, since xarray works with hypercubes it cannot
#   handle these differnt types of levels.
#   open_datasets handles this creating an xarray Dataset  per level-type and storing
#   all these different Datasets in a list
fcn = cfgrib.open_datasets("output/fourcastnet/fourcastnet-20170115.grib")

# Extract all Dataset from list for easier handling
tcwv = fcn[0]       # total column water vapour
uv10m = fcn[1]      # 10m u and v wind
t2m = fcn[2]        # 2m temperature
uv100m = fcn[3]     # 100m u and v wind
z = fcn[4]          # upper-air geopotential (5 pressure levels)
t = fcn[5]          # upper-air temperature (3 pressure levels)
uv = fcn[6]         # upper-air u and v wind (4 pressure levels)
r = fcn[7]          # relative humidity (2 pressure levels)
mslp = fcn[8]       # mean sea level pressure
sp = fcn[9]         # surface pressure
pcp = fcn[10]       # accumulated total precipitation


# Let's manipulate some data
# - Lets have look at the upper-air temperature
t
# The Dataset conains:
# - 1 data variable(DataArray): t (a bit confusing naming, I know)

# Let's extract this DataArray
t.t
t['t']
# --> both command are identical

# This DataArray has 6 coordinates from which 4 are main dimensions (they have a * next to them):
# - time: startime of the forecast
t.t.time
# - step: the progression of the forecast (in nanoseconds)
t.t.step  
# --> if you want to show them in a more readable format you can use
t.t.step.values.astype('timedelta64[h]') # in hours

# - isobaricInhPa: the pressure level (in hPa)
t.t.isobaricInhPa
# - longitude and latitude in (degrees)
t.t.longitude
t.t.latitude
# - valid_time: the date for which the forecast is valid
t.t.valid_time


# Now lets extract a certain timestep, and pressure level
# - We can do this on the basis of the value of the timestep, and level e.g. +12h, 850hPa
t.t.sel(
    step=np.timedelta64(12,"h"),
    isobaricInhPa=850.0
)

# - Or we can select on the basis of there respecitve indices in the array i.e. 2 and 0
t.t.isel(
    step=2,
    isobaricInhPa=0
)

# Now let's visualize this
t.t.isel(
    step=2,
    isobaricInhPa=0
).plot(
    x="longitude",
    y="latitude"
)
# use plt.show() to show the image
# This is not very nice-looking, lets pimp our plot
fig = plt.figure(figsize=(16,9))
ax = plt.subplot(projection=ccrs.Robinson()) # Visually apealing projection
selection = t.t.isel(
    step=2,
    isobaricInhPa=0
)
selection.plot.contourf(
    x="longitude",
    y="latitude",
    ax = ax,
    cmap = "plasma",                # Change the colormap
    robust=True,                    # Takes care of outliers in colorbar
    levels=15,                      # Number of levels for the contours
    transform=ccrs.PlateCarree()    # Projection of the data (lat-lon)
)
ax.coastlines()
nice_title = f'FourCastNet T{int(selection.isobaricInhPa.values)} forecast for' + \
    f' {np.datetime_as_string(selection.time.values,"s")}' + \
    f' +{selection.step.values.astype("timedelta64[h]")}' + \
    f'\n valid at {np.datetime_as_string(selection.valid_time.values,"s")}'
plt.title(nice_title)




# AIFS
aifs = cfgrib.open_datasets("output/aifs/aifs-20170115.grib")

tcw = aifs[0]           # total column water
uv10m = aifs[1]         # 10m u and v wind
td2m = aifs[2]          # 2m temperature and dewpoint-temperature
upper_air = aifs[3]     # upper air variables (z,t,u,v,q,w) at 13 pressure levels
mslp = aifs[4]          # mean sea level pressure
sfc_cst = aifs[5]       # constant surface variables (not forecasted)
pcp = aifs[6]           # accumulated convective and total precipitation
sfc = aifs[7]           # surface variables (sp=surface pressure, skt=skin/surface temperature)

# AIFS data is a bit different to work with because the data is on a reduced gaussian grid
# For more information: https://confluence.ecmwf.int/display/FCST/Gaussian+grids
# In practive this means that the data does not have a lon and lat dimension
# In stead each gridpoint is given a id-value, and for each value the lon and lat are given 
t2m = td2m.t2m
t2m.longitude
t2m.latitude

# Unlike regular lat-lon grids, the lat-lon coordinates do not from a nice matrix, 
# i.e. for a given latitude, different longitude values exist.

# We need a small workaround if we want to make nice plot
# First we need to 'unstack' the value dimension to lat-lon coordinates
t2m['values'] = pd.MultiIndex.from_arrays(
    [t2m.longitude.values,t2m.latitude.values], 
    names=['longitude', 'latitude']
)
t2m = t2m.unstack('values')

# Since xarray works with hypercubes, each lat must have the same longitude values and vice versa
# This is not the case for the AIFS data, so there are a lot of NaNs in the data,
t2m

# We can fix this by interpolating from the neighbouring points that do have data

t2m = t2m.interpolate_na(
    "longitude",
    "cubic",
    fill_value="extrapolate"
).interpolate_na(
    "latitude",
    "cubic",
    fill_value="extrapolate"
)

# Let's plot
fig = plt.figure(figsize=(16,9))
ax = plt.subplot(projection=ccrs.Robinson()) # Visually appealing projection
selection = t2m.isel(step=2)
selection.plot.contourf(
    x="longitude",
    y="latitude",
    ax = ax,
    cmap = "plasma",                # Change the colormap
    robust=True,                    # Takes care of outliers in colorbar
    levels=15,                      # Number of levels for the contours
    transform=ccrs.PlateCarree()    # Projection of the data (lat-lon)
)
ax.coastlines()
nice_title = f'AIFS T{int(selection.heightAboveGround.values)}m forecast for' + \
    f' {np.datetime_as_string(selection.time.values,"s")}' + \
    f' +{selection.step.values.astype("timedelta64[h]")}' + \
    f'\n valid at {np.datetime_as_string(selection.valid_time.values,"s")}'
plt.title(nice_title)

