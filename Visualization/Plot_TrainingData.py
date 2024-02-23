import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

columns=['loss', 'floss', 'closs', 'tloss', 'FCEnum', 'volume']
# df = pd.DataFrame(columns=['loss', 'floss', 'closs', 'tloss', 'vloss', 'FCEnum', 'volume'])
# df = pd.read_csv("linearSat_initV1.csv", usecols=columns)

# obs training
# columns=['loss', 'floss', 'closs', 'tloss', 'FCEnum', 'volume']
# df = pd.read_csv("loss_epoch0.csv", usecols=columns)

# linear sat
columns=['loss', 'floss', 'closs', 'tloss', 'vloss', 'FCEnum', 'volume']
df = pd.read_csv("linearSat_fixn.csv", usecols=columns)
# df1 = pd.read_csv("loss_epoch1_.csv", usecols=columns)
fig = pl.figure(figsize=[7,5], tight_layout=True)
pl.plot(df.loss, label='loss', linewidth=4)
# pl.plot(df.vloss+df.floss+df.closs+df.tloss, label='loss', linewidth=4)
pl.plot(df.floss, label='feasibility', linewidth=4)
pl.plot(df.closs, '--', label='correctness', linewidth=4)
# pl.plot(df.volume, label='volume')
pl.xlabel('Epoch', fontsize=20)
pl.ylabel('Training Loss', fontsize=20)
pl.xlim([0, 19])
pl.xticks(np.arange(0, 19, step=2), fontsize=20)
pl.yticks(fontsize=20)
pl.legend(fontsize=20)
plt.title('FT-NCBF training loss along epoch', fontsize=20)

pl.show()
