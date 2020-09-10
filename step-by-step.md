# Step by step

#### Data Generation
The scripts in this folder produce the required datasets for pix2pix.
For now, all functionalities are to be edited within the script.
*To Do: Merge scripts into one with multiple functionalities, add arguments, develop input formats*
- gen_data_balanced produces a dataset where the wind directions are balanced according to a ratio specified within script. If there are not enough samples for a direction in a certain month to fulfill the ratio, that month will have less samples than specified. The input format (eg wind, pressure) can be modified, the output is the footprint with the reference red dot.

- gen_data_noiseless produces the same dataset as above, only the footprint in the output shows only the higher concentration areas (all points below 4*average are removed)
- gen_data_simultaneous combines both scripts above - produces one set of inputs and two sets of outputs, one with noise and one without noise

The scripts output the images, of approximately the selected size, to an specified folder, that should contain two folders inside (A and B, or A, B_noise and B_noiseless in the simultaneous one) each with train, test and val folders inside. Changing the folder name in the `create_directories.txt` (or the `create_simultaneous directories.txt`) , then executing them with `bash create_directories.txt`, creates the necessary folders and subfolders.

#### Databases
Once the data is generated, it is combined so it is usable by pix2pix. This is done using 
```
python /path/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A /data_generation/new_set/A --fold_B /data_generation/new_set/B --fold_AB /databases/new_set
```
It requires the Python module.
*To do: add sample databases*

#### Training
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
`--size`| For unet256, it only supports images whose width and height are divisible by 256. For unet128, the width and height need to be divisible by 128. For resnet_6blocks and resnet_9blocks, the width and height need to be divisible by 4.
`--lambda_L1` | Modifies the lambda parameter in the loss function (pix2pix's loss function is cGAN_loss + lambda*L1_loss).

See the pix2pix [options file](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options) for other base, training and testing options.

 ##### Visualising the results
As pix2pix trains, it saves an example at each epoch to track development. All finished epochs can be accessed in the `checkpoints/jobname/web`folder. It is recomended that the whole folder is scp'd to the local computer and the html opened in a standard browser. (Note: it is recommended that before opening the html file in the local machine, it is edited to remove `http-equiv="refresh"` from line five. This command makes the html refresh every few seconds, which is not useful if it has been scp'd).
The loss plots can be visualised using the `plot_loss.py` file (functional but needs developing)

**Training in bluepebble**
pix2pix requires a high amount of memory (depending on  the size of the dataset, more fine-tuning needed) and about 2.5 hours to run, depending on the size of the dataset. See [the sample pbs file](pix2pix_gpu_sample.pbs).

#### Testing
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

Note that pictures need to have a certain size during training (square, with a size determined by the network architecture), but they do not need this during testing. Thus, images bigger than the specified size during training / different ratio etc can be passed.

##### Visualising the results
Once the testing is done, you can visualise the results by visualising `index.html`in the folder `results/jobname/test_latest`. It is recomended that the whole folder is scp'd to the local computer and the html opened in a standard browser.

The results can be evaluated and plotted using the `compare_results.py` file (functional but needs developing).