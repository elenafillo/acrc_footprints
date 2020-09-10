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
"""

path = "/work/ef17148/acrc_footprints/data_generation/wind_to_fp_balanced_big/"

pathA = {"train" : path + "A/train/", "test" : path + "A/test/", "val" : path + "A/val/"}
pathB = {"train" : path + "B/train/", "test" : path + "B/test/", "val" : path + "B/val/"}

# number of samples in each folder per month - will be selected evenly spaced throughout the year
ntrain = 150
nval = 20
ntest = 20
nsamples = {"train" : ntrain, "test" : ntest, "val" : nval}

""" size of image (cut to square centered around release point). must be even and divisible by 256
if using unet256. if using resnet_6blocks and resnet_9blocks, must be divisible by four.
the original images are cut to a square of sizexsize images centered around the release point (if possible) but matplotplib cannot save pictures of an exact pixel size (only inch size) so current code produces images 
of size around 350x350 that are cut when processed by pix2pix.
"""
size = 256

# min ratio of each wind direction, to avoid a very imbalanced dataset. must add up to one
ratios = {"west":0.3, "north":0.2, "east" : 0.2, "south":0.3}

# path to metfiles
path_x = "/work/al18242/ML_summer_2020/EUROPE_Met_slp_"
#path to footprints
path_y = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_"
# filenames
filenames = [201801 + i for i in range(12)]

# variables - all variables = ["Temperature", "Pressure", "PBLH", "Wind_Speed", "Wind_Direction"]
# can select only some to test differences
variables = ["Wind_Direction", "Wind_Speed"]

# release point is identified and size around it determined, only done once then init_size = False (see below)
init_size = False

save = False


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
            # need to do s = s+1 somewhere below!!!
            
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
            if  (wind_dir < 45 or  wind_dir > 315) and to_save_samples["north"] > 0:
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
                LON, LAT = np.meshgrid(fp.lon, fp.lat)

                fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
                ax.contourf(LON, LAT, np.log10(fp),levels=51)
                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red")
                ax.axis('off')
                plt.savefig(pathB[mode]+str(time)[:13]+".jpg",pad_inches=0, bbox_inches = "tight", dpi=166)

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
                if "Pressure" in variables:
                    cs = ax.contourf(LON, LAT, met_data_cut["Pressure"],levels=51)
                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red")
                ax.axis('off')
                plt.savefig(pathA[mode]+str(time)[:13]+".jpg", bbox_inches='tight',pad_inches = 0,  dpi=166)            
                plt.close("all")
            s = s + 1
           
            if s == len(timestamps):
                print("!!!!all timestamps checked but could not complete ratio", to_save_samples)
                for d in to_save_samples.keys():
                    to_save_samples[d] = 0 
