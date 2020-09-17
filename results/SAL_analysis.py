import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import random
import os
import math
import argparse
from scipy import ndimage

"""
This script analyses the quality of the results produced by pix2pix using the SAL measure (https://journals.ametsoc.org/mwr/article/136/11/4470/68138/SAL-A-Novel-Quality-Measure-for-the-Verification)

for each job, it returns a histogram of each of the SAL components (explained belowin the calculate_sal function) and three plots showing how the three components interact
"""

# Initialise parser and define arguments
parser = argparse.ArgumentParser(description='generate pix2pix dataset from netcdf files')
parser.add_argument('--jobname', required=True, nargs='*', help='jobname to evaluate or jobnames to evaluate and compare')
args = parser.parse_args()

def calculate_sal(fake, real):
    # note - fake and real should be in greyscale (0-1 white-black)
    
    # AMPLITUDE provides a simple measure of the quantitative accuracy of the total amount of precipitation in a specified region D, ignoring the field’s subregional structure
    # A ranges [-2, 2] where 0 is a perfect forecast, positive values indicate an overestimation and viceversa
    
    def D(R):
        N = np.shape(R)[0] * np.shape(R)[1]
        return np.sum(R)*(1/N)
    
    AMP = (D(fake)-D(real))/(0.5*(D(fake)+D(real)))
    
    # LOCATION measures the normalized distance between the centers of mass of the modeled and observed fields
    # L ranges [0,1] where 0 is perfect overlap of the centers of mass and 1 is the maximum distance.
    def x(R):
        return ndimage.measurements.center_of_mass(R)
    
    # d is the largest distance of two gridpoints in domain -i.e. the diagonal
    d = math.sqrt(2*(np.shape(real)**2))
    LOC = math.sqrt((x(fake)[0] - x(real)[0])**2 + (x(fake)[1] - x(real)[1])**2)/d
    
    # STRUCTURE compares the volume of the normalized objects
    # S ranges [-2, 2] where 0 is a perfect prediction
    
    def V(R):
        return np.sum(R/np.max(R))
    
    STRUC = (V(fake)-V(real))/(0.5*(V(fake)+V(real)))
    
    return STRUC, AMP, LOC 
    
 
def rgb2gray(rgb):
    # greyscale goes from 0 (black) to 1 (white)
    grey =  np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    # it is inverted so it goes from 0 (white) to 1 (black)
    return abs(grey - 1)
    
for job in args.jobname:
    print(job)
    all_images = os.listdir( job  + '/test_latest/images')
    # for each date, theres an input (real_A) a prediction (fake_B) and the true output (real_B) - so we take each name only once
    
    # results col 0 is S, col 1 is A and col 2 is L
    results = np.zeros((int(len(all_images)/3),3))
    n = 0
    for im in all_images[::3]:
        # print percentage as progress
        if n%20 == 0:
            print(round(100*n/int(len(all_images)/3)),"%")          
        # take the date from the name as a string
        date = im[:13]
        
        # load the images
        fake = image.imread("./" + job + '/test_latest/images/' + date + '_fake_B.png')
        real = image.imread("./" + job + '/test_latest/images/' + date + '_real_B.png')
        # transform them to inverse grayscale
        fake = rgb2gray(fake)
        real = rgb2gray(real)    
        
        S,A,L = calculate_sal(fake, real)
        results[n,0] = S
        results[n,1] = A
        results[n,2] = L
        
    fig_hist, (axS, axA, axL) = plt.subplots(1,3, figsize(15,20))
    axS.hist(results[:,0], 15)
    axA.hist(results[:,1], 15)
    axL.hist(results[:,2], 15)
    
    axS.set_title("S")
    axA.set_title("A")
    axL.set_title("L")
    fig_hist.suptitle("Histograms of SAL values for" + job)
# do histogram for all samples?
# identify images with highest lowest and check
 
 