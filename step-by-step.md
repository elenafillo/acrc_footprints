# Step by step

#### Data Generation
The scripts in this folder produce the required datasets for pix2pix.
For now, all functionalities are to be edited within the script.
*To Do: Merge scripts into one with multiple functionalities, add arguments, develop input formats*
- gen_data_balanced produces a dataset where the wind directions are balanced according to a ratio specified within script. If there are not enough samples for a direction in a certain month to fulfill the ratio, that month will have less samples than specified. The input format (eg wind, pressure) can be modified, the output is the footprint with the reference red dot.

- gen_data_noiseless produces the same dataset as above, only the footprint in the output shows only the higher concentration areas (all points below 4*average are removed)
- gen_data_simultaneous combines both scripts above - produces one set of inputs and two sets of outputs, one with noise and one without noise

The scripts output the images, of approximately the selected size, to an specified folder, that should contain two folders inside (A and B, or A, B_noise and B_noiseless in the simultaneous one) each with train, test and val folders inside. Changing the folder name in the `create_directories.txt` (or the `create_simultaneous directories.txt`) , then executing them with `bash create_directories.txt`, creates the necessary folders and subfolders.
