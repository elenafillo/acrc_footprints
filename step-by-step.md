# Step by step

## Data Generation
The script `generate_data.py` in this folder produces the required datasets for pix2pix. It requires the pyhton module to run. It is executed by running the following command (see optional flags below):
```
python -W ignore generate_data.py --dataroot new_set 
```
The script takes the metereological and footprint .nc files (located within bluepebble, for a different location the code should be changed), prepares the output and input according to the flags and saves it to an output folder. The script can function in three ways:
- Input to footprint (default): input is prepared according to flags, and the output is the footprint
- Input to footprint (noiseless): input is prepared according to flags, and the output is the points of highest concentration of the footprint (noise has been removed). Activated with the flag `--noiseless`
- Simultaneous: input is prepared according to flags, and both the default and the noiseless footprints are returned to parallel folders within the output folder. Activated with the flag `--simultaneous`.

#### Output folder
The output folder should contain two folders inside (A and B, or A, B_noise and B_noiseless in the simultaneous one) each with train, test and val folders inside. Changing the folder name in the `create_directories.txt` (or the `create_simultaneous_directories.txt`) , then executing them with `bash create_directories.txt`, creates the necessary folders and subfolders.

#### Balance in the dataset
The `--ratios` flag allows the user to control the balance of the dataset with respect to wind direction at the release point by specifying a given ratio for each direction (north, east, south, west), adding up to one. For example, a perfectly balanced dataset would be inputted via `--ratios 0.25 0.25 0.25 0.25`, and a dataset containing only west-directed samples would be achieved with  `--ratios 0 0 0 1`. The script aims to get as close as possible to the desired ratio, but in cases where the original distribution of directions in the data is very uneven, it will not be successful and will return less samples than expected. For example, aiming for a completely balanced dataset where the data has barely any east samples will result in the correct number of north, west and south samples but few east samples. 

#### Sizing
The input data is cropped to a square around the release point of size (size_cut, size_cut), inputted with the flag `--size_cut`. This is the equivalent of "zooming in" the full image. Make sure that size_cut is less than half the height and the width.

The cropped image is outputted with approximately the pixel size specified by `--cut_output`. Choosing a value close to that accepted by the preferred generator architecture will work best. For example, unet256 (the default architecture in pix2pix) supports images whose width and height in pixels are multiples of 256. Due to the nature of matplotlib, used to save images, the exact `--cut_output` size cannot be achieved, so an image slightly above that size is returned and cut to the correct size during training. 

#### Script options
This is the role of some relevant flags that allow input and output customisation:
Flag | Role
------------ | -------------
`--dataroot` | folder to save the generated data in. Must have been created before running the program, using `bash create_directories.txt` (or `bash create_simultaneous_directories.txt` if the `--simultaneous` flag is used.)
`--simultaneous`| Activates the simultaneous mode, where both the default and noiseless footprints are returned.
`--noiseless`| Returns noiseless footprints instead of default.
`--ntrain` | Number of train samples to generate (may not be completed depending on the ratios and the nature of the data). int, default = 600.
`--ntest` | Number of test samples to generate (may not be completed depending on the ratios and the nature of the data). int, default = 100.
`--nval` | Number of val samples to generate (may not be completed depending on the ratios and the nature of the data). int, default = 100.
`--size_cut`| Size to cut datafile to, centered around release point. See Sizing. int, default = 64.
`--size_output`| Approx size of output figure. Allowed sizes depend on generator architecture. See Sizing. int, default = 256
`--ratios`| Ratio of wind direction samples in dataset. Must be four numbers adding up to one. four floats, default = 0.2 0.2 0.3 0.3
`--wind_speed`| Input plot - contoured plot of wind speed (51 levels)
`--wind_direction`| Input plot - quiver plot of wind direction 
`--sea_level_pressure`| Input plot - coloured plot of pressure (5 levels)
`--direction_at_release`| Input plot - arrow in the direction of the windspeed at the release point, length dependent on speed.
`--size_red_dot`| Size of the red dot located at the release point. int, default = 10, a value of 0 means no red dot is drawn.

## Databases
Once the data is generated, it is combined so it is usable by pix2pix. This is done using 
```
python /path/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A data_generation/new_set/A --fold_B data_generation/new_set/B --fold_AB databases/new_set
```
It requires the Python module.

To download one of the sample databases, do
```
bash download_dataset.sh datasetname
```
See [the summary](summary_and_future_steps.md) for the sample datasets available.

## Training
Training and testing require the pytorch module, and if you want to use GPU to run it, then the cuda module, added with `module load lang/cuda`. 

Once loaded, `cd`to `pytorch-CycleGAN-and-pix2pix` and execute the following:
```
python train.py --dataroot /path/acrc_footprints/databases/database_name --name jobname --model pix2pix --direction AtoB --gpu_ids 0,1,2 --display_id -1 --preprocess crop --crop_size 256 --netG unet_256 --checkpoints_dir /path/acrc_footprints/checkpoints

```
This is the role of some relevant flags:
Flag | Role
------------ | -------------
`--dataroot` | dataset to train on, combined using `combine_A_and_B.py`
`--name` | Name of the job. Checkpoints will be saved in a folder of this name under the directory specified by `--checkpoints_dir`.
`--gpu_ids` | pix2pix can run on CPU (value of -1) or on up to three GPUs (0,1,2 respectively). The cuda module is required except for -1.
`--display_id` | The training results and plot loss can be loaded on a local web server as training happens. -1 disables this.
`--netG` | Defines the generator architecture (unet256, unet128, resnet_6blocks and resnet_9blocks)
`--preprocess` | pix2pix requires square images for training, and the size depends on the generator architecture. See the [pix2pix tips doc](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/300e84a78e77e22f08668180c65949971386175b/docs/tips.md) for different types of preprocess.
`--crop_size`| For unet256, it only supports images whose width and height are divisible by 256. For unet128, the width and height need to be divisible by 128. For resnet_6blocks and resnet_9blocks, the width and height need to be divisible by 4.
`--lambda_L1` | Modifies the lambda parameter in the loss function (pix2pix's loss function is cGAN_loss + lambda*L1_loss).

See the pix2pix [options file](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options) for other base, training and testing options.

#### Note on sizes:
Using the flag `--preprocess crop` during training means the input image is randomly cut to a square of size `--crop_size` (during testing, images of any size can be passed). The red dot has the function of indicating the position of the release point, rather than passing correctly-sized images where the release point would always be centered. This way, it is likely that the trained network can produce footprints for any release location and not only the centre (needs more testing). 

#### Sample checkpoints
To download one of the sample checkpoint files, use
```
bash download_checkpoints.sh jobname
```
See [the summary](summary_and_future_steps.md) for the sample datasets available.

#### Visualising the results
As pix2pix trains, it saves an example at each epoch to track development. All finished epochs can be accessed in the `checkpoints/jobname/web`folder. It is recomended that the whole folder is scp'd to the local computer and the html opened in a standard browser. (Note: it is recommended that before opening the html file in the local machine, it is edited to remove `http-equiv="refresh"` from line five. This command makes the html refresh every few seconds, which is not useful if it has been scp'd).
The loss plots can be visualised using the `plot_loss.py` file, running
```
python plot_loss.py --name jobname
```
It requires the python module.

**Training in bluepebble**
pix2pix requires a high amount of memory (depending on  the size of the dataset, more fine-tuning needed) and about 2.5 hours to run, depending on the size of the dataset. See [the sample pbs file](pix2pix_gpu_sample.pbs).

## Testing
Testing of small datasets can be conducted directly on the terminal without needing to submit it to bluepebble. It requires the pytorch module.
It can be executed with the following command (from the `pytorch-CycleGAN-and-pix2pix`):
```
python test.py --dataroot /path/acrc_footprints/databases/database_name --name jobname --model pix2pix --direction AtoB --gpu_ids -1  --checkpoints_dir /path/acrc_footprints/checkpoints --results_dir /work/ef17148/acrc_footprints/results 
``` 
This is the role of some relevant flags (some overlap with training):
Flag | Role
------------ | -------------
`--name`|  Name of the job. pix2pix will use the checkpoints stored in the folder under the jobname, and will return the results to a new folder under the same name.
`--num_test` | Number of samples to test from the test folder under name. Default is 50. If num_test is bigger than the number of pictures in the folder, it just tests them all.
`--netG` | Same generator architecture as used in training


Note that pictures need to have a certain size during training (square, with a size determined by the network architecture), but they do not need this during testing. Thus, images bigger than the specified size during training / different ratio etc can be passed.
#### Sample results
To download one of the sample result files, use
```
bash download_dataset.sh jobname
```
See [the summary](summary_and_future_steps.md) for the sample datasets available.

#### Visualising the results
Once the testing is done, you can visualise the results by visualising `index.html`in the folder `results/jobname/test_latest`. It is recomended that the whole folder is scp'd to the local computer and the html opened in a standard browser.

The results can be evaluated and plotted using the `plot_results.py`file (functional but needs developing). It can be executed using
```
python -W ignore plot_results.py --jobname JOBNAME(S)
```
A single jobname will return a graph of the sum of the predicted footprint vs the real footprint, whereas multiple jobnames will return the above for each plus a comparative graph of the mean error and the correctly guessed pixels. It is roughly orientative (and badly coded), and a better error measure like SAL should be developed.

The results can also be evaluated using the [SAL measure](https://journals.ametsoc.org/mwr/article/136/11/4470/68138/SAL-A-Novel-Quality-Measure-for-the-Verification), which originally evaluates how accurate a rainfall prediction is compared to the actual event. It is used here to evaluate the accuracy of the predicted footprint, returning three key markers as follow:
- **Structure S**: compares the volume of the normalised objects. It is assumed in this use that there's only one object present in each frame. It can take values in the [-2, 2] range where 0 is a perfect forecast, positive values indicate an overestimation of the spread and negative values appear for too small or too peaked objects.
- **Amplitude A**: provides a simple measure of the quantitative accuracy of the total amount of concentration in the frame, ignoring the subregional structure. It can take values in the [-2, 2] range, where 0 is a perfect forecast, positive values indicate an overestimation and viceversa.
- **Location L**: measures the normalised distance between the center of mass of the modeled and observed fields. It can take values in the [0,1] range where 0 is a perfect overlap and 1 is the maximum possible distance in the frame.
Please refer to the [SAL paper](https://journals.ametsoc.org/mwr/article/136/11/4470/68138/SAL-A-Novel-Quality-Measure-for-the-Verification) for further clarification.

The script can be executed using
```
python -W ignore SAL_analysis.py --jobname JOBNAME(S)
```
For each jobname introduced (separated by spaces), the script calculates the SAL measure for every image and returns the average while plotting a histogram of each component.