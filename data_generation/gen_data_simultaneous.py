import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random


"""
this script generates a dataset usable in the pix2pix algorithm,
and stores it in two separate folders in the specified loc (see requirements)
Requirements:
    - image size must be divisible by 256
    - folder /path/to/data must subdirectories A and B, each with one image style.
      Both must have subdirectories train, val, test.
    - Corresponding images in a pair {A,B} must be the same size and have the same
    filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

use the comand bash create_directories.txt in the folder where you want to generate A and B
to create a new folder called "gen_data" with A, B etc inside.

Select variables to be used as input.

Checks the wind direction at the release point and provides a balanced dataset according to the provided ratios
removes all points that are below 4*mean value (very similar script to balanced data)
"""

path = "/work/ef17148/acrc_footprints/data_generation/press_arrow_to_fp_both/"

pathA = {"train" : path + "A/train/", "test" : path + "A/test/", "val" : path + "A/val/"}
pathB_noise = {"train" : path + "B_noise/train/", "test" : path + "B_noise/test/", "val" : path + "B_noise/val/"}
pathB_noiseless = {"train" : path + "B_noiseless/train/", "test" : path + "B_noiseless/test/", "val" : path + "B_noiseless/val/"}


# number of samples in each folder per month - will be selected evenly spaced throughout the year
ntrain = 60
nval = 10
ntest = 10
nsamples = {"train" : ntrain, "test" : ntest, "val" : nval}

""" size of image (cut to square centered around release point). must be even and divisible by 256
if using unet256. if using resnet_6blocks and resnet_9blocks, must be divisible by four.
matplotplib cannot save pictures of an exact pixel size (only inch size) so current code produces images 
of size around 350x350 that are cut when processed by pix2pix.
"""
size = 256

# min ratio of each wind direction, to avoid a very imbalanced dataset. must add up to one
ratios = {"west":0.3, "north": 0.20, "east" : 0.20, "south":0.3}

# path to metfiles
path_x = "/work/al18242/ML_summer_2020/EUROPE_Met_slp_"
#path to footprints
path_y = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_"
# filenames
filenames = [201801 + i for i in range(12)]

# variables - all variables = ["Temperature", "Pressure", "PBLH", "Wind_Speed", "Wind_Direction"]
# can select only some to test differences
variables = ["Sea_level_pressure"]

# release point is identified and size around it determined, only done once then init_size = False (see below)
init_size = False


for month in filenames:
    met_data = xr.open_dataset(path_x + str(month) + ".nc")

    fp_data = xr.open_dataset(path_y + str(month) + ".nc")

    for mode in ["train", "test", "val"]:
        # randomly order all the possible timestamps
        timestamps = random.sample(range(0, len(met_data.time.values)), len(met_data.time.values))

        
        to_save_samples = {orientation:num*nsamples[mode] for orientation, num in ratios.items()}
        print(to_save_samples)
        s = 0
        """ for each timestamp (flagged by s), the direction of the wind is found. If there are more samples with that 
        orientantion required, it is saved, otherwise s moves up to try the next timestamp. this stops when there are enough samples. 
        """
        # for each datapoint
    
        while np.any(np.array(list(to_save_samples.values()))>0):
            print("mode", mode, "month", month-201801, "item", s)
            print(to_save_samples)
            time = met_data.time.values[timestamps[s]]
            
            save = False
            
            
            # find location of release point so can find the cutting boundaries
            # (assumed that release point is the same throughout the dataset), only conducted once
            if init_size == False:
                fp_o = fp_data.fp.sel({"time":pd.to_datetime(time)})
                met = met_data.sel({"time":pd.to_datetime(time)})
                # find latitude-longitude datapoints closest to the release coordinates
                release_lat = min(met.lat.values, key=lambda x:abs(x-fp_data.release_lat[0]))
                release_lon = min(met.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))
                
                       
                release_lon_fp = min(fp_data.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))
                # find the index of these datapoints and determine boundaries
                lat_bound =[int(np.where(fp_o.lat.values == release_lat)[0][0]-size/2), int(np.where(fp_o.lat.values == release_lat)[0][0]+size/2)]
                lon_bound =[int(np.where(fp_o.lon.values == release_lon_fp)[0][0]-size/2), int(np.where(fp_o.lon.values == release_lon_fp)[0][0]+size/2)]

                # check if boundaries are outside of image size
                if lat_bound[0] < 0:
                    lat_bound[0] = 0
                    lat_bound[1] = size - 1
                if lat_bound[1] >= fp_o.values.shape[0]:
                    lat_bound[1] = fp_o.values.shape[0] - 1
                    lat_bound[0] = fp_o.values.shape[0] - 1 - size
                if lon_bound[0] < 0:
                    lon_bound[0] = 0
                    lon_bound[1] = size -1
                if lon_bound[1] >= fp_o.values.shape[1]:
                    lon_bound[1] = fp_o.values.shape[1] - 1
                    lon_bound[0] = fp_o.values.shape[1] - 1 - size

                print(lat_bound, lon_bound)
                # change to true so boundaries are not recalculated
                init_size = True


            # put here some checks to see if there is space blalblaalba
            
            print("Checking wind direction of current file")
            met_data_cut = met_data.sel(time = time, lat = slice(met_data.lat.values[lat_bound[0]], met_data.lat.values[lat_bound[1]]),
                lon = slice(met_data.lon.values[lon_bound[0]], met_data.lon.values[lon_bound[1]]))
             
            wind_dir = met_data_cut["Wind_Direction"].sel({"lat" : release_lat, "lon" : release_lon}).values
            if  (wind_dir < 45 or wind_dir > 315) and to_save_samples["north"] > 0:
                to_save_samples["north"] = to_save_samples["north"] - 1 
                save = True
            if wind_dir < 135 and wind_dir > 45 and to_save_samples["east"] > 0:
                to_save_samples["east"] = to_save_samples["east"] -1
                save = True
            if wind_dir < 225 and wind_dir > 135 and to_save_samples["south"] > 0:
                to_save_samples["south"] = to_save_samples["south"] - 1
                save = True
            if wind_dir < 315 and wind_dir > 225 and to_save_samples["west"] > 0:
                to_save_samples["west"] = to_save_samples["west"] - 1 
                save = True
            
            
            
            if save == True:
            


                print("     saving footprint file ", str(time)[:13],".jpg")
                # create footprint image and save
                fp = fp_data.fp.sel(time = time, lat = slice(fp_o.lat.values[lat_bound[0]], fp_o.lat.values[lat_bound[1]]),
                    lon = slice(fp_o.lon.values[lon_bound[0]], fp_o.lon.values[lon_bound[1]]))
                vals = fp.values
                LON, LAT = np.meshgrid(fp.lon, fp.lat)
                fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
                ax.contourf(LON, LAT, np.log10(vals),levels=51)
                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=10)
                ax.axis('off')
                plt.savefig(pathB_noise[mode]+str(time)[:13]+".jpg",pad_inches=0, bbox_inches = "tight", dpi=166)

                vals[vals< 4*np.mean(vals)] = 0

                fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
                ax.contourf(LON, LAT, np.log10(vals),levels=51)
                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=10)
                ax.axis('off')
                plt.savefig(pathB_noiseless[mode]+str(time)[:13]+".jpg",pad_inches=0, bbox_inches = "tight", dpi=166)

                print("     saving met file ", str(time)[:13],".jpg")
                # create meteorological image and save
                LON, LAT = np.meshgrid(met_data_cut.lon, met_data_cut.lat)
                fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
                if "Wind_Speed" in variables:
                    wind_to_plot = met_data_cut["Wind_Speed"]
                    cs = ax.contourf(LON, LAT, wind_to_plot,levels=51)
                if "Wind_Direction" in variables:
                    ax.quiver(LON[::10, ::10], LAT[::10, ::10],
                         np.cos(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                         np.sin(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                         angles='xy', scale_units='xy', scale=0.5)
                if "Sea_level_pressure" in variables:
                    cs = ax.contourf(LON, LAT, met_data_cut["Sea_level_pressure"],levels=5)

                    ax.quiver(fp_data.release_lon[0],  fp_data.release_lat[0], np.sin(2*np.pi/360.*wind_dir),
                         np.cos(2*np.pi/360.*wind_dir), angles = 'xy',scale = 1/met_data_cut["Wind_Speed"].sel({"lat" : release_lat, "lon" : release_lon}).values, scale_units = 'xy')

                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=10)
                ax.axis('off')
                plt.savefig(pathA[mode]+str(time)[:13]+".jpg", bbox_inches='tight',pad_inches = 0,  dpi=166)            
                plt.close('all')

            s = s + 1
            
            if s == round(0.75*len(timestamps)):
                print("75% of timestamps checked, could not fullfil ratio. Images needed:", to_save_samples)
                for direction in to_save_samples.keys():
                    to_save_samples[direction] = 0
