import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import random
import os
import math


def rgb2gray(rgb):
    # greyscale goes from 0 (black) to 1 (white)
    grey =  np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    # it is inverted so it goes from 0 (white) to 1 (black)
    return abs(grey - 1)


path = "/work/al18242/ML_summer_2020/EUROPE_Met_slp_"
filenames = [201801 + i for i in range(12)]
all_paths = [path + str(fn) + '.nc' for fn in filenames]
met_data_full = xr.open_mfdataset(all_paths, combine="by_coords")

results_path = '/work/ef17148/acrc_footprints/results/'
results = ['press_arrow_to_fp_noise','press_arrow_to_fp_noiseless', 'press_arrow_to_fp_noise_biglambda', 'press_arrow_to_fp_noiseless_biglambda', 'press_arrow_to_fp_noise_smalllambda',  'press_arrow_to_fp_noiseless_smalllambda', 'press_arrow_to_fp_noise_nolambda', 'press_arrow_to_fp_noiseless_nolambda', 'press_arrow_to_fp_noise_verybiglambda', 'press_arrow_to_fp_noiseless_verybiglambda' ]
#results = ['press_arrow_to_fp_noise']


# get release coordinates
footprint_file = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_201801.nc"
fp_data = xr.open_dataset(footprint_file)
# find latitude-longitude datapoints closest to the release coordinates
release_lat = min(met_data_full.lat.values, key=lambda x:abs(x-fp_data.release_lat[0]))
release_lon = min(met_data_full.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))


emissions_map = xr.open_dataset("/work/al18242/ML_summer_2020/ch4_EUROPE_2013.nc")
flux = emissions_map.flux.values[36:292,122:378,0]
# cut to dimensions

fig, ax = plt.subplots()
#fig2, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(results)/2))
fig3, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, nrows = 1)

xaxis = 0
for dataset in results:
    print(dataset)
    all_images = os.listdir(results_path + dataset  + '/test_latest/images')
    # for each date, theres an input (real_A) a prediction (fake_B) and the true output (real_B)

    # results - col 0 is wind direction, col 1 is np.sum(abs(real - fake)), col 2 is np.sum(real) - np.sum(fake), 
    # col 3 is % of correct pixels (how many pixels were correctly identified as zero (white in both pictures) or nonzero (some colour in both pictures))
    # col 4 is % of pixels in real footprint that were correctly identified as such
    # col 5 is % of pixels outside of the real footprint that were correctly identified as such
    results_dir = np.zeros((int(len(all_images)/3), 6))
    # sums - col 0 is sum(real) and col 1 is sum(fake)
    sums = np.zeros((int(len(all_images)/3),2))
    # sums_emissions - col 0 is sum(flux*real), col 1 is sum(flux*fake)
    sums_emissions = np.zeros((int(len(all_images)/3),2))

    n = 0


    for im in all_images[::3]:
        date = im[:13]

        fake = image.imread(results_path + dataset + '/test_latest/images/' + date + '_fake_B.png')
        fake = rgb2gray(fake)
        real = image.imread(results_path + dataset + '/test_latest/images/' + date + '_real_B.png')
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

        fake[fake > 0.1] = 1
        fake[fake < 0.1] = 0
        real[real > 0.1] = 1
        real[real < 0.1] = 0
        results_dir[n,3] = 100*np.sum(fake == real)/(256*256)
        results_dir[n,4] = 100*np.sum(fake[real == 1] == real[real == 1])/np.sum(real == 1)
        results_dir[n,5] = 100*np.sum(fake[real == 0] == real[real == 0])/np.sum(real == 0)

        wind_dir = met_data_full["Wind_Direction"].sel({"time":pd.to_datetime(date), "lat" : release_lat, "lon" : release_lon}).values
        results_dir[n,0] = wind_dir

        n = n + 1

    print(dataset)
    # average error

    ax1.scatter(xaxis, np.mean(results_dir[:,3]), label = dataset[18:])
    ax2.scatter(xaxis, np.mean(results_dir[:,4]))
    ax3.scatter(xaxis, np.mean(results_dir[:,5]))

    print("average total error:", np.mean(results_dir[:,1]), "std", np.std(results_dir[:,1]))
    ax.scatter(xaxis, np.mean(results_dir[:,1]), c = 'k')

    mask_below = results_dir[:,0] < 45
    mask_above = results_dir[:,0] > 315
    joint = results_dir[np.logical_or(mask_below, mask_above), 1]
    print("NORTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    ax.scatter(xaxis+0.2, np.mean(joint), c = 'b')

    mask_below = results_dir[:,0] > 45
    mask_above = results_dir[:,0] < 135
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("EAST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    ax.scatter(xaxis+0.4, np.mean(joint), c = 'y')

    mask_below = results_dir[:,0] > 135
    mask_above = results_dir[:,0] < 225
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("SOUTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    ax.scatter(xaxis+0.6, np.mean(joint), c = 'm')

    mask_below = results_dir[:,0] > 225
    mask_above = results_dir[:,0] < 315
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("WEST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    ax.scatter(xaxis+0.8, np.mean(joint), c = 'g')

    xaxis = xaxis + 1
    fig2, new_ax = plt.subplots()
    new_ax.scatter(sums_emissions[:,0], sums_emissions[:,1])
    new_ax.plot([sums_emissions.min(), sums_emissions.max()], [sums_emissions.min(), sums_emissions.max()])
    new_ax.set_title(dataset)

    fig2.suptitle("sum of predicted*emissions versus sum of real*emissions")
    fig2.tight_layout(pad = 2.0)


fig3.suptitle("% of correctly identified pixels (as white or non-white)")
ax1.set_title("overall")
ax2.set_title("in the footprint")
ax3.set_title("outside the footprint")
handles, labels = ax1.get_legend_handles_labels()
fig3.legend(handles, labels, loc = 'center right')
ax4.axis('off')

ax.set_xticks(np.arange(len(results))+0.5)
ax.set_xticklabels(results, rotation = 35, ha = "right")

plt.show()
