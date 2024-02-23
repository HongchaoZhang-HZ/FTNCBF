import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

dfs = pd.read_csv("trajr_6d_succ.csv")
dff = pd.read_csv("trajr_6d_fail.csv")
# df1 = pd.read_csv("loss_epoch1_.csv", usecols=columns)
# fig = plt.figure(figsize=[8,5])
fig, ax = plt.subplots(figsize=[10,5.5], tight_layout=True)
ax.plot(dff['value'], 'crimson', label='Baseline', linewidth=4)
ax.plot(dfs['value'], 'g', label='Proposed Approach', linewidth=4)
ax.add_patch(Rectangle((0,1.5),50,0.25, color='salmon', label='Unsafe Region'))
ax.add_patch(Rectangle((0,0),50,0.25, color='salmon'))
# pl.plot(df.volume, label='volume')
plt.xlabel('Time-steps', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Distance to target satellite', fontsize=20)
plt.text(17, 1.55, 'Unsafe Region', color='brown', fontsize=30)
plt.text(17, 0.05, 'Unsafe Region', color='brown', fontsize=30)
plt.xlim([0, 50])
plt.ylim([0, 1.75])
# pl.xticks(np.arange(0, 19, step=2))
plt.legend(fontsize=20, loc=(0.01, 0.2))
plt.title('Trajectory comparison between baseline and proposed method', fontsize=20)
plt.show()
