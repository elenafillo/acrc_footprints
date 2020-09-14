# Summary and future steps
This document contains a summary of the runs attempted so far, together with some thoughts on their effectiveness. Moreover, below there is a list of possible tests and improvements.

## Experiments conducted
- Datasets *wind to footprint* - input: wind speed and direction; output: footprint. Both cut a square of 256 pixels. With noise it recognises the overall shape but does not identify areas of high concentration, with the noiseless dataset it has difficulty extracting the main direction of the plume. 
- Datasets *pressure to footprint* - input: pressure coloured contour plot, arrow with the direction of the wind at release point with length depending on the windspeed; output: footprint (see in the databases folder, with noise `press_to_fp_noise` and without `press_to_fp_noiseless`). Both cut a square of 256 pixels. Different lambda values were tested (see [notes](README.md)):
	- no lambda: returns the same prediction for any input (both noise and noiseless) (lambda = 0)
	- small lambda: only two/three fixed outcomes depending on the input (lambda = 10)
	- default lambda: For noise dataset, locates the area of the cloud relatively well but does not distinguish areas of higher and lower concentration. For the noiseless dataset, returns a very similar looking shape for all inputs. See the checkpoints and the results under the name `press_to_fp_noise_lambda100` and `press_to_fp_noiseless_lambda100`.(lambda = 100)
	- big lambda: For the noise dataset, relatively good prediction of the cloud shape and some attempt for colour. For the noiseless, three main types of returns - thin footprint west, thin footprint east or no footprint. The direction is not always correct.(lambda = 1000) 
	- very big lambda: For noise dataset, good prediction of the cloud shape but complete loss of colour (this is common for L1). Scatter plot of the sum of predicted vs sum of real approaches a diagonal shape. For the noiseless, some samples are well predicted but others just show blurry results or indistinctive shapes.  See the checkpoints and the results under the name `press_to_fp_noise_lambda5000` and `press_to_fp_noiseless_lambda5000`.(lambda = 5000) 
These results make sense, given than small lambdas give more importance to "returning goodlooking footprints" and high lambdas give more importance to spacially accurate results but ignore colours and sharp edges.

Further investigation focuses on only noiseless datasets.
- Dataset *press_to_fp_noiseless_small* - same input and output as  *press_to_fp_noiseless*, but cut to a square of 64 pixels instead (zoomed in) (see in the databases folder as `press_to_fp_noiseless_small`). Red circle had size 2 (instead of default size 10). The images outputted have a size of around 256 pixels. Again, different lambda values were tested:
	- default lambda: Overall directionality often correct but not necessarily shape (eg curve vs straight).  See the checkpoints and the results under the name `press_to_fp_noiseless_small_lambda100`.(lambda = 100)
	- big lambda: Both directionality and shape often correct. Some cases of blurriness/undefined edges and colours, or disjointed footprints (eg white spaces between pieces that should be connected).  See the checkpoints and the results under the name `press_to_fp_noiseless_small_lambda1000`.(lambda = 1000).
	- very big lambda: Both directionality and shape often correct but strong blurriness/undefined edges and colours in most predictions, and many present disjointed footprints. See the checkpoints and the results under the name `press_to_fp_noiseless_small_lambda10000`.(lambda = 10000).

- Datasets *footprint to future footprint* - input: footprint at time t with quiver on top, output: footprint at time t+3 and t+6. Investigation not followed as less relevant to goal. Most footprints are too similar at t and t+6 to detect relevant changes, so pix2pix only removes quiver.



## Tests and improvements

#### Not changing the original pix2pix
- Improving error measure and comparison (eg implement SAL)
- Testing further lambda values for the small dataset (parameter tuning, probably between default and big lambda)
- Testing known lambda values with more epochs
- Testing different cutting sizes 
- Different input sets, smaller red dot (although pressure + arrow seems to work well)
- Further parameter tuning - more/less generator and discriminator layers


#### Changing pix2pix
- Making SAL the error function
- 