import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
plt.plot(np.random.randn(50).cumsum(), 'k--')
ax2 = fig.add_subplot(2, 2, 2)
plt.plot(np.random.randn(50).cumsum(), 'k--')
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(np.random.randn(50).cumsum(), 'k--')

plt.legend()
strFile = "/Users/frishman/Downloads/pl.png"
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)

strFile = "/Users/frishman/Downloads/pl2.png"
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
