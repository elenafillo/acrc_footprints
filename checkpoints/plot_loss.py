import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('press_arrow_to_fp_noise_smalllambda/loss_log.txt', sep=" ", skiprows = 1)#, skiprows = [0,1968,1699, 1700, 1701, 1702, 1703, 1704, 1705],header=None)

data.rename(columns = {data.columns[1]:'epoch',data.columns[3]:'iters', data.columns[5]:'time', data.columns[7]:'data', data.columns[9]:'G_GAN', data.columns[11]:'G_L1', data.columns[13]:'D_real', data.columns[15]:'D_fake'}, inplace = True)
data.drop(data.columns[0], axis = 1)
data.drop(columns = [data.columns[0], data.columns[2], data.columns[4], data.columns[6], data.columns[8], data.columns[10], data.columns[12], data.columns[14]])
print(len(data))
fig, axes = plt.subplots(nrows = 2, ncols = 2)

data.plot(y = 'G_GAN', ax = axes[0,0])
data.plot(y = 'G_L1', ax = axes[0,1])
data.plot(y = 'D_real', ax = axes[1,0])
data.plot(y = 'D_fake', ax = axes[1,1])

plt.suptitle("press + arrow to fp, noise dataset ")
plt.show()
