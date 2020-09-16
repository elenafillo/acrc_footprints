import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import random
import os
import math
import argparse

"""
This script plots the results of training and testing with the pix2pix algorithm. It returns the following:
    - if only one job added, it evaluates that job by printing the average error overall + in each wind direction, and it plots sum(flux*real) vs sum(flux*fake) (if size_cut == size_output), or sum(real) vs sum(fake) otherwise.
    - if more than one job is added, it returns the above for each job plus two comparative graphs:
        - % of correctly identified pixels (as white or non white)
        - average total error overall and in each direction
"""

# Initialise parser and define arguments
parser = argparse.ArgumentParser(description='generate pix2pix dataset from netcdf files')


parser.add_argument('--jobname', required=True, nargs='*', help='jobname to evaluate or jobnames to evaluate and compare')

parser.add_argument('--size_cut', type=int, default = 256, help = 'Size file was cut to during data generation')
parser.add_argument('--size_output', type=int, default = 256, help = 'Size of output figure inputted during data generation. If size_cut is not equal to size output, the flux information will not be used.')




def rgb2gray(rgb):
    # greyscale goes from 0 (black) to 1 (white)
    grey =  np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    # it is inverted so it goes from 0 (white) to 1 (black)
    return abs(grey - 1)
    
    
def get_flux(args.size_cut):
    # get release coordinates
    met_data = xr.open_dataset("/work/al18242/ML_summer_2020/EUROPE_Met_slp_201801.nc")
    fp_data = xr.open_dataset("/work/al18242/ML_summer_2020/MHD-10magl_EUROPE_201801.nc")
    # find latitude-longitude datapoints in metdata closest to the release coordinates
    release_lat = min(met_data_full.lat.values, key=lambda x:abs(x-fp_data.release_lat[0]))
    release_lon = min(met_data_full.lon.values, key=lambda x:abs(x-fp_data.release_lon[0]))  
    # get emissions map
    emissions_map = xr.open_dataset("/work/al18242/ML_summer_2020/ch4_EUROPE_2013.nc")
    # find the index of these datapoints and determine boundaries
    lat_bound =[int(np.where(fp_data.lat.values == release_lat)[0][0]-args.size_cut/2), int(np.where(fp_data.lat.values == release_lat)[0][0]+args.size_cut/2)]
    lon_bound =[int(np.where(fp_data.lon.values == release_lon_fp)[0][0]-args.size_cut/2), int(np.where(fp_data.lon.values == release_lon_fp)[0][0]+args.size_cut/2)]
    
    # cut to dimensions
    flux = emissions_map.flux.values[lon_bound[0]:lon_bound[1], lat_bound[0]:lat_bound[1],0]
    
    return flux

print(len(args.jobname))

if len(args.jobname) > 1:  #compare
    compare == True
    #fig shows the average total error for each wind direction
    fig, ax = plt.subplots()
    #fig3 shows the % of correctly identified pixels
    fig3, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, nrows = 1)
    xaxis = 0
else:
    compare == False
    
for job in args.jobname:
    print(dataset)
    all_images = os.listdir(results_path + job  + '/test_latest/images')
    # for each date, theres an input (real_A) a prediction (fake_B) and the true output (real_B) - so we take each name only once
       
    
    """
    pre-define result arrays
    results_dir - col 0 is wind direction, col 1 is np.sum(abs(real - fake)), col 2 is np.sum(real) - np.sum(fake), col 3 is % of correct pixels (how many pixels were correctly identified as zero (white in both pictures) or nonzero (some colour in both pictures)), col 4 is % of pixels in real footprint that were correctly identified as such, col 5 is % of pixels outside of the real footprint that were correctly identified as such

    if size_cut == size, output
    sums col 0 is sum(flux*real), col 1 is sum(flux*fake)
    
    otherwise, 
    # sums  - col 0 is sum real, col 1 is sum fake 
    """
    sums = np.zeros((int(len(all_images)/3),2))
    results = np.zeros((int(len(all_images)/3), 3))
    n = 0

    for im in all_images[::3]:
        # print percentage as progress
        if n%10 == 0:
            print(round(100*n/int(len(all_images)/3)),"%")
            
        # take the date from the name as a string
        date = im[:13]
        
        # load the images
        fake = image.imread(results_path + dataset + '/test_latest/images/' + date + '_fake_B.png')
        real = image.imread(results_path + dataset + '/test_latest/images/' + date + '_real_B.png')
        # transform them to inverse grayscale
        fake = rgb2gray(fake)
        real = rgb2gray(real)
        
        # store the sum (including flux or not)
        if args.size_cut == args.size_output:
            #get flux cut to the same domain
            flux = get_flux(args.size_cut)
            sums[n,0] = 1e9*np.sum(flux * real)
            sums[n,1] = 1e9*np.sum(flux * fake)
        else:
            sums[n,0] = np.sum(real)
            sums[n,1] = np.sum(fake)
            
        
         
        # transform images to black (1) and white (0)
        fake[fake > 0.1] = 1
        fake[fake < 0.1] = 0
        real[real > 0.1] = 1
        real[real < 0.1] = 0
        # count how many pixels are correctly identified
            # overall
        results_dir[n,3] = 100*np.sum(fake == real)/(args.size_output*args.size_output)
            # within the real footprint 
        results_dir[n,4] = 100*np.sum(fake[real == 1] == real[real == 1])/np.sum(real == 1)
            # outside of the real footprint
        results_dir[n,5] = 100*np.sum(fake[real == 0] == real[real == 0])/np.sum(real == 0)
        
        # store wind direction
        wind_dir = met_data_full["Wind_Direction"].sel({"time":pd.to_datetime(date), "lat" : release_lat, "lon" : release_lon}).values
        results_dir[n,0] = wind_dir

        n = n + 1
    
    # print and plot results
    print(job)
    
    # fig2 is sum(real) vs sum(predicted), with or without flux
    fig2, new_ax = plt.subplots()
    new_ax.scatter(sums[:,0], sums[:,1])
    new_ax.plot([sums.min(), sums.max()], [sums.min(), sums.max()])
    new_ax.set_title(dataset)
    if args.size_cut == args.size_output:
        fig2.suptitle("sum of predicted*emissions versus sum of real*emissions")
    else:
        fig2.suptitle("sum of predicted versus sum of real")
    fig2.tight_layout(pad = 2.0)

    
    print("average total error:", np.mean(results_dir[:,1]), "std", np.std(results_dir[:,1]))
    
    if compare == True:
        # scatter percentages of correctly identified pixels
        ax1.scatter(xaxis, np.mean(results_dir[:,3]), label = dataset[18:])
        ax2.scatter(xaxis, np.mean(results_dir[:,4]))
        ax3.scatter(xaxis, np.mean(results_dir[:,5]))
        
        # print average results and scatter 
        ax.scatter(xaxis, np.mean(results_dir[:,1]), c = 'k')

     
    mask_below = results_dir[:,0] < 45
    mask_above = results_dir[:,0] > 315
    joint = results_dir[np.logical_or(mask_below, mask_above), 1]
    print("NORTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    if compare == True:
        ax.scatter(xaxis+0.2, np.mean(joint), c = 'b')

    mask_below = results_dir[:,0] > 45
    mask_above = results_dir[:,0] < 135
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("EAST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    if compare == True:
        ax.scatter(xaxis+0.4, np.mean(joint), c = 'y')

    mask_below = results_dir[:,0] > 135
    mask_above = results_dir[:,0] < 225
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("SOUTH:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    if compare == True:
        ax.scatter(xaxis+0.6, np.mean(joint), c = 'm')

    mask_below = results_dir[:,0] > 225
    mask_above = results_dir[:,0] < 315
    joint = results_dir[np.logical_and(mask_below, mask_above), 1]
    print("WEST:", len(joint), "items, avg error", np.mean(joint), "stdev", np.std(joint))
    if compare == True:
        ax.scatter(xaxis+0.8, np.mean(joint), c = 'g')
        xaxis = xaxis + 1
        
        
if compare == True:
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
    



    
