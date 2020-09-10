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

This script creates an image with a footprint and wind at a given time, and the resulting footprint three hrs later
"""

path = "/work/ef17148/acrc/data/gen_data_overlap/"

pathA = {"train" : path + "A/train/", "test" : path + "A/test/", "val" : path + "A/val/"}
pathB = {"train" : path + "B/train/", "test" : path + "B/test/", "val" : path + "B/val/"}

# number of samples in each folder per month - will be selected evenly spaced throughout the year
ntrain = 70
nval = 12
ntest = 12
nsamples = {"train" : ntrain, "test" : ntest, "val" : nval}

# size of image (cut to square centered around release point). must be even and divisible by 256
# if using unet256. if using resnet_6blocks and resnet_9blocks, must be divisible by four.
size = 256


# path to metfiles 
path_x = "/work/al18242/ML_summer_2020/EUROPE_Met_"
#path to footprints
path_y = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_"
# filenames
filenames = [201801 + i for i in range(12)]

# variables - all variables = ["Temperature", "Pressure", "PBLH", "Wind_Speed", "Wind_Direction"]
# can select only some to test differences
variables = ["Wind_Speed", "Wind_Direction"]

# release point is identified and size around it determined, only done once then init_size = False (see below)
init_size = False


for month in filenames:
    met_data = xr.open_dataset(path_x + str(month) + ".nc")
    fp_data = xr.open_dataset(path_y + str(month) + ".nc")

    for mode in ["train", "test", "val"]:
        # generate random numbers in the range of available datapoints, select those datapoints
        timestamps = random.sample(range(0, len(met_data.time.values)-3), nsamples[mode])

        # for each datapoint generated
        for s in range(nsamples[mode]):
            print("mode", mode, "month", month-201801, "item", s)
            time = met_data.time.values[timestamps[s]]
            next_time = met_data.time.values[timestamps[s]+3]

            # get footprint data at the given time
            fp = fp_data.fp.sel({"time":pd.to_datetime(time)})

            # find location of release point so can find the cutting boundaries
            # (assumed that release point is the same throughout the dataset), only conducted once
            if init_size == False:
                # find latitude-longitude datapoints closest to the release coordinates
                release_lat = min(fp.lat.values, key=lambda x:abs(x-fp_data.release_lat[0]))
                release_lon = min(fp.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))
                # find the index of these datapoints and determine boundaries
                lat_bound =[int(np.where(fp.lat.values == release_lat)[0][0]-size/2), int(np.where(fp.lat.values == release_lat)[0][0]+size/2)]
                lon_bound =[int(np.where(fp.lon.values == release_lon)[0][0]-size/2), int(np.where(fp.lon.values == release_lon)[0][0]+size/2)]
                
                # check if boundaries are outside of image size
                if lat_bound[0] < 0:
                    lat_bound[0] = 0
                    lat_bound[1] = size - 1
                if lat_bound[1] >= fp.values.shape[0]:
                    lat_bound[1] = fp.values.shape[0] - 1
                    lat_bound[0] = fp.values.shape[0] - 1 - size
                if lon_bound[0] < 0:
                    lon_bound[0] = 0
                    lon_bound[1] = size -1 
                if lon_bound[1] >= fp.values.shape[1]:
                    lon_bound[1] = fp.values.shape[1] - 1  
                    lon_bound[0] = fp.values.shape[1] - 1 - size
                 
                print(lat_bound, lon_bound)
                # change to true so boundaries are not recalculated
                init_size = True
            
            
            print("     saving footprint file ", str(time)[:13],".jpg", "time", str(next_time)[:13])  
            # create footprint image and save
            fp_next = fp_data.fp.sel(time = next_time, lat = slice(fp.lat.values[lat_bound[0]], fp.lat.values[lat_bound[1]]),
                lon = slice(fp.lon.values[lon_bound[0]], fp.lon.values[lon_bound[1]]))
            LON, LAT = np.meshgrid(fp_next.lon, fp_next.lat)
            #here we take log10 of the footprints due to the exponential drop off of sensitivity

            fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
            ax.contourf(LON, LAT, np.log10(fp_next),levels=51)   
            ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red")     
            ax.axis('off')            
            plt.savefig(pathB[mode]+str(time)[:13]+".jpg", pad_inches = 0, bbox_inches = "tight", dpi=166)
            
            print("     saving met file ", str(time)[:13],".jpg")
            # create meteorological image and save
            fp = fp_data.fp.sel(time = time, lat = slice(fp.lat.values[lat_bound[0]], fp.lat.values[lat_bound[1]]),
                lon = slice(fp.lon.values[lon_bound[0]], fp.lon.values[lon_bound[1]]))
            met_data_cut = met_data.sel(time = time, lat = slice(met_data.lat.values[lat_bound[0]], met_data.lat.values[lat_bound[1]]),
                lon = slice(met_data.lon.values[lon_bound[0]], met_data.lon.values[lon_bound[1]]))
            LON, LAT = np.meshgrid(met_data_cut.lon, met_data_cut.lat)
            fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=166)
            ax.contourf(LON, LAT, np.log10(fp),levels=51) 
            ax.quiver(LON[::10, ::10], LAT[::10, ::10],
                 np.cos(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                 np.sin(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                 angles='xy', scale_units='xy', scale=0.5)
                 
            ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red")
            ax.axis('off')    
            plt.savefig(pathA[mode]+str(time)[:13]+".jpg", pad_inches = 0, bbox_inches='tight', dpi=166)
