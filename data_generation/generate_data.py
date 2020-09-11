import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse


"""
this script generates a dataset usable in the pix2pix algorithm,
and stores it in the required format in the specified path.

Before running it, edit the create_directories.txt (or the create_simultaneous_directories.txt) to your desired foldername,
and then use the comand bash create_directories.txt to generate the required folders.

See required and optional arguments below for customisation, or read the section on Data Generation in the step_by_step.doc.

TODO: add argument to add path data to met and footprints file
"""

""" size of image (cut to square centered around release point). must be even and divisible by 256
if using unet256. if using resnet_6blocks and resnet_9blocks, must be divisible by four.
matplotplib cannot save pictures of an exact pixel size (only inch size) so current code produces images 
of size around 350x350 that are cut when processed by pix2pix.
"""



# Initialise parser and define arguments
parser = argparse.ArgumentParser(description='generate pix2pix dataset from netcdf files')

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
group = parser.add_mutually_exclusive_group() # simultaneous and noiseless are mutually exclusive. neither means noise output only
group.add_argument('--simultaneous', action= 'store_true', help = 'generate two simultaneous datasets (noise and noiseless)')
group.add_argument('--noiseless', action = 'store_true', help = 'output footprint is noiseless. Do not use combined with --simultaneous')
parser.add_argument('--ntrain', type=int, default = 600, help = 'number of train samples (may not be completed depending on ratios)')
parser.add_argument('--ntest', type=int, default = 100, help = 'number of test samples (may not be completed depending on ratios)')
parser.add_argument('--nval', type=int, default = 100, help = 'number of val samples (may not be completed depending on ratios)')
parser.add_argument('--size_cut', type=int, default = 64, help = 'Size to cut file to, centered around release point.')
parser.add_argument('--size_output', type=int, default = 256, help = 'Approx size of output figure. Allowed sizes depend on architecture but are not checked! Please check before running')
parser.add_argument("--ratios", nargs=4, metavar=('north', 'east', 'south', 'west'), help="Ratio of wind directions in datasets. Must be floats adding up to one", type=float, default=[0.2, 0.2, 0.3, 0.3])
parser.add_argument("--wind_speed", action = 'store_true', help = 'coloured contour plot of wind speed (51 levels)')
parser.add_argument("--wind_direction", action = 'store_true', help = 'quiver plot of wind direction')
parser.add_argument("--sea_level_pressure", action = 'store_true', help = 'coloured contour plot of pressure (5 levels)')
parser.add_argument("--direction_at_release", action = 'store_true', help = 'add arrow in the direction of the windspeed at the release point')
parser.add_argument("--size_red_dot", type = int, default = 10, help = 'size of red dot at release point (size 0 means no dot')

args = parser.parse_args()


# retrieve the met and footprints data 
path_x = "/work/al18242/ML_summer_2020/EUROPE_Met_slp_" # path to metfiles
path_y = "/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_" #path to footprints
filenames = [201801 + i for i in range(12)] # create list of paths to each file
filenames_x = [path_x + str(filen) + '.nc' for filen in filenames]
filenames_y = [path_y + str(filen) + '.nc' for filen in filenames] 
met_data = xr.open_mfdataset(filenames_x, combine="by_coords") # open all met and fp data combined 
fp_data = xr.open_mfdataset(filenames_y, combine="by_coords")




# prepare the storing paths
pathA = {"train" : args.dataroot + "/A/train/", "test" : args.dataroot + "/A/test/", "val" : args.dataroot + "/A/val/"}
if args.simultaneous == True:
    # if simultaneous, path B contains B_noise
    pathB= {"train" : args.dataroot + "/B_noise/train/", "test" : args.dataroot + "/B_noise/test/", "val" : args.dataroot + "/B_noise/val/"}
    pathB_noiseless = {"train" : args.dataroot + "/B_noiseless/train/", "test" : args.dataroot + "/B_noiseless/test/", "val" : args.dataroot + "/B_noiseless/val/"}
else:
    pathB = {"train" : args.dataroot + "/B/train/", "test" : args.dataroot + "/B/test/", "val" : args.dataroot + "/B/val/"}


# number of samples in each folder 
nsamples = {"train" : args.ntrain, "test" : args.ntest, "val" : args.nval}

# ratios
ratios = {"north": args.ratios[0], "east" : args.ratios[1], "south": args.ratios[2], "west": args.ratios[3]}



# variables - all variables = ["Temperature", "Pressure", "PBLH", "Wind_Speed", "Wind_Direction"]
# can select only some to test differences
variables = ["Sea_level_pressure"]

# release point is identified and size around it determined, only done once then init_size = False (see below)
init_size = False



for mode in ["train", "test", "val"]:
    # randomly order all the possible timestamps
    timestamps = random.sample(range(0, len(met_data.time.values)), len(met_data.time.values))

    
    to_save_samples = {orientation:num*nsamples[mode] for orientation, num in ratios.items()}
    print(to_save_samples)
    s = 0
    """ for each timestamp (flagged by s), the direction of the wind is found. If there are samples with that 
    orientantion required, it is saved, otherwise s moves up to try the next timestamp. this stops when there are enough samples. 
    """
    # for each datapoint

    while np.any(np.array(list(to_save_samples.values()))>0):
        print("mode", mode, "item", s)
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
            lat_bound =[int(np.where(fp_o.lat.values == release_lat)[0][0]-args.size_cut/2), int(np.where(fp_o.lat.values == release_lat)[0][0]+args.size_cut/2)]
            lon_bound =[int(np.where(fp_o.lon.values == release_lon_fp)[0][0]-args.size_cut/2), int(np.where(fp_o.lon.values == release_lon_fp)[0][0]+args.size_cut/2)]

            # check if boundaries are outside of image size
            if lat_bound[0] < 0:
                lat_bound[0] = 0
                lat_bound[1] = arg.size_cut - 1
            if lat_bound[1] >= fp_o.values.shape[0]:
                lat_bound[1] = fp_o.values.shape[0] - 1
                lat_bound[0] = fp_o.values.shape[0] - 1 - args.size_cut
            if lon_bound[0] < 0:
                lon_bound[0] = 0
                lon_bound[1] = args.size -1
            if lon_bound[1] >= fp_o.values.shape[1]:
                lon_bound[1] = fp_o.values.shape[1] - 1
                lon_bound[0] = fp_o.values.shape[1] - 1 - args.size_cut

            print(lat_bound, lon_bound)
            # change to true so boundaries are not recalculated
            init_size = True


        
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
            fig, ax = plt.subplots(figsize=(args.size_output/100, args.size_output/100), dpi=166)
            
            if args.simultaneous == False and args.noiseless == True:
                vals[vals< 4*np.mean(vals)] = 0 #saves only noiseless one
            
            ax.contourf(LON, LAT, np.log10(vals),levels=51)
            ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=args.size_red_dot)
            ax.axis('off')
            plt.savefig(pathB[mode]+str(time)[:13]+".jpg",pad_inches=0, bbox_inches = "tight", dpi=166)

            if args.simultaneous == True:
                vals[vals< 4*np.mean(vals)] = 0 #saves noiseless one as well
                fig, ax = plt.subplots(figsize=(args.size_output/100, args.size_output/100), dpi=166)
                ax.contourf(LON, LAT, np.log10(vals),levels=51)
                ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=args.size_red_dot)
                ax.axis('off')
                plt.savefig(pathB_noiseless[mode]+str(time)[:13]+".jpg",pad_inches=0, bbox_inches = "tight", dpi=166)

            print("     saving met file ", str(time)[:13],".jpg")
            # create meteorological image and save
            LON, LAT = np.meshgrid(met_data_cut.lon, met_data_cut.lat)
            fig, ax = plt.subplots(figsize=(args.size_output/100, args.size_output/100), dpi=166)
            
            if args.wind_speed == True:
                wind_to_plot = met_data_cut["Wind_Speed"]
                cs = ax.contourf(LON, LAT, wind_to_plot,levels=51)
                
            if args.wind_direction == True:
                ax.quiver(LON[::10, ::10], LAT[::10, ::10],
                     np.cos(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                     np.sin(3*np.pi/2-2*np.pi/360.*met_data_cut["Wind_Direction"])[::10, ::10],
                     angles='xy', scale_units='xy', scale=0.5)
            if args.sea_level_pressure == True:
                cs = ax.contourf(LON, LAT, met_data_cut["Sea_level_pressure"],levels=5)
            if args.direction_at_release == True:
                ax.quiver(fp_data.release_lon[0],  fp_data.release_lat[0], np.sin(2*np.pi/360.*wind_dir),
                     np.cos(2*np.pi/360.*wind_dir), angles = 'xy',scale = 1/met_data_cut["Wind_Speed"].sel({"lat" : release_lat, "lon" : release_lon}).values, scale_units = 'xy')
          
            ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red", s=args.size_red_dot)
            ax.axis('off')
            plt.savefig(pathA[mode]+str(time)[:13]+".jpg", bbox_inches='tight',pad_inches = 0,  dpi=166)            
            plt.close('all')

        s = s + 1
        
        if s == round(0.9*len(timestamps)):
            print("90% of timestamps checked, could not fullfil ratio. Images still needed:", to_save_samples)
            for direction in to_save_samples.keys():
                to_save_samples[direction] = 0
