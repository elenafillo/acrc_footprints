import pandas as pd
from matplotlib import pyplot as plt

"""
This script generates a plot of the training loss for the training logs of the pix2pix algorithm.

Run after the training has been completed.
"""

parser = argparse.ArgumentParser(description='plot training loss')

parser.add_argument('--jobname', required = True, help = "Name of the training job to plot loss of.")

args = parser.parse_args()

data = pd.read_csv(args.jobname + '/loss_log.txt', sep=" ", skiprows = 1)

data.rename(columns = {data.columns[1]:'epoch',data.columns[3]:'iters', data.columns[5]:'time', data.columns[7]:'data', data.columns[9]:'G_GAN', data.columns[11]:'G_L1', data.columns[13]:'D_real', data.columns[15]:'D_fake'}, inplace = True)
data.drop(data.columns[0], axis = 1)
data.drop(columns = [data.columns[0], data.columns[2], data.columns[4], data.columns[6], data.columns[8], data.columns[10], data.columns[12], data.columns[14]])
print("iterations":len(data))
fig, axes = plt.subplots(nrows = 2, ncols = 2)

data.plot(y = 'G_GAN', ax = axes[0,0])
data.plot(y = 'G_L1', ax = axes[0,1])
data.plot(y = 'D_real', ax = axes[1,0])
data.plot(y = 'D_fake', ax = axes[1,1])

plt.suptitle(args.jobname)
plt.show()
