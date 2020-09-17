import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import random
import os
import math
from scipy import ndimage

def calculate_sal(fake, real):
    # note - fake and real should be in greyscale (0-1 white-black)
    
    # AMPLITUDE provides a simple measure of the quantitative accuracy of the total amount of precipitation in a specified region D, ignoring the field’s subregional structure
    
    def D(R):
        N = np.shape(R)[0] * np.shape(R)[1]
        return np.sum(R)*(1/N)
    
    AMP = (D(fake)-D(real))/(0.5*(D(fake)+D(real)))
    
    # LOCATION measures the normalized distance between the centers of mass of the modeled and observed fields
    
    def x(R):
        return ndimage.measurements.center_of_mass(R)
    
    # d is the largest distance of two gridpoints in domain -i.e. the diagonal
    d = math.sqrt(2*(np.shape(real)**2))
    LOC = math.sqrt((x(fake)[0] - x(real)[0])**2 + (x(fake)[1] - x(real)[1])**2)/d
    
    # STRUCTURE compares the volume of the normalized objects
    
    def V(R):
        return np.sum(R/np.max(R))
    
    STRUC = (V(fake)-V(real))/(0.5*(V(fake)+V(real)))
    
    return [AMP, LOC, STRUC]
        