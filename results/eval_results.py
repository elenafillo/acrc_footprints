import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import random
import os


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    
path = "/work/al18242/ML_summer_2020/EUROPE_Met_slp_"
filenames = [201801 + i for i in range(12)]
all_paths = [path + str(fn) + '.nc' for fn in filenames]
met_data_full = xr.open_mfdataset(all_paths, combine="by_coords")

results_path = '/work/ef17148/acrc_footprints/results/met_to_fp_balanced'
all_images = os.listdir(results_path + '/test_latest/images')
# for each date, theres an input (real_A) a prediction (fake_B) and the true output (real_B)


# get release coordinates
footprint_file = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_201801.nc"
fp_data = xr.open_dataset(footprint_file)
# find latitude-longitude datapoints closest to the release coordinates
release_lat = min(met_data_full.lat.values, key=lambda x:abs(x-fp_data.release_lat[0]))
release_lon = min(met_data_full.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))


# get emissions map
emissions_map = xr.open_dataset("/work/al18242/ML_summer_2020/ch4_EUROPE_2013.nc")
flux = emissions_map.flux.values[36:292,122:378,0]
# cut to dimensions


results_dir = np.zeros((int(len(all_images)/3), 3))
sums = np.zeros((int(len(all_images)/3),2))
sums_emissions = np.zeros((int(len(all_images)/3),2))
n = 0
for im in all_images[::3]:
    date = im[:13]

    fake = image.imread(results_path + '/test_latest/images/' + date + '_fake_B.png')
    fake = rgb2gray(fake)
    real = image.imread(results_path + '/test_latest/images/' + date + '_real_B.png')
    real = rgb2gray(real)
    diff_total = np.sum(real) - np.sum(fake)
    diff = np.sum(abs(real - fake))
    sums[n,0] = np.sum(real)
    sums[n,1] = np.sum(fake)
    sums_emissions[n,0] = 1e9*np.sum(flux * real)
    sums_emissions[n,1] = 1e9*np.sum(flux * fake)
    if n%10 == 0:
        print(round(100*n/int(len(all_images)/3)),"%")
    results_dir[n,1] = diff
    results_dir[n,2] = diff_total


    wind_dir = met_data_full["Wind_Direction"].sel({"time":pd.to_datetime(date), "lat" : release_lat, "lon" : release_lon}).values
    results_dir[n,0] = wind_dir
    #if  wind_dir < 45 and wind_dir > 315: #north
    #    results[n,0] = 1
    #if wind_dir < 135 and wind_dir > 45: #east
    #    results[n,0] = 2
    #if wind_dir < 225 and wind_dir > 135: #south
    #    results[n,0] = 3
    #if wind_dir < 315 and wind_dir > 225: #west
    #    results[n,0] = 4
    n = n + 1


# average error
print("average total error:", np.mean(results_dir[:,1]), "std", np.std(results_dir[:,1]))

mask_below = results_dir[:,0] < 45
mask_above = results_dir[:,0] > 315
joint = results_dir[np.logical_or(mask_below, mask_above), 1]
print("NORTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))

mask_below = results_dir[:,0] > 45
mask_above = results_dir[:,0] < 135
joint = results_dir[np.logical_and(mask_below, mask_above), 1]
print("EAST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))

mask_below = results_dir[:,0] > 135
mask_above = results_dir[:,0] < 225
joint = results_dir[np.logical_and(mask_below, mask_above), 1]
print("SOUTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))

mask_below = results_dir[:,0] > 225
mask_above = results_dir[:,0] < 315
joint = results_dir[np.logical_and(mask_below, mask_above), 1]
print("WEST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))


fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(results_dir[:,0], results_dir[:,1])
ax2.scatter(results_dir[:,0], results_dir[:,2])
ax1.set_title("sum(abs(real - fake))")
ax2.set_title("sum(a) - sum(b)")
fig.suptitle("difference in pixels between real and predicted \n depending on wind direction")
for xcoord in [45, 135, 225, 315]:
    ax1.axvline(x = xcoord)
    ax2.axvline(x = xcoord)
ax2.axhline(y=0)
plt.tight_layout()
plt.subplots_adjust(top=0.85)




fig, ax = plt.subplots()
ax.scatter(sums[:,0], sums[:,1])
fig.suptitle("sum of predicted versus sum of real")
plt.plot([sums.min(), sums.max()], [sums.min(), sums.max()])
plt.tight_layout()

fig, ax = plt.subplots()
ax.scatter(sums_emissions[:,0], sums_emissions[:,1])
fig.suptitle("sum of predicted*emissions versus sum of real*emissions")
plt.plot([sums_emissions.min(), sums_emissions.max()], [sums_emissions.min(), sums_emissions.max()])
plt.tight_layout()


plt.show()

