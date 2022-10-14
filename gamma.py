import sys
from learn import Learn
import torch
import numpy as np
import pandas as pd
import random

pos = torch.load("/Users/frishman/Dropbox/Bioinformatics/projects/embed/secretase/t5/pos.t5.pt")
neg = torch.load("/Users/frishman/Dropbox/Bioinformatics/projects/embed/secretase/t5/neg.t5.pt")

embed = []
lab = []
proteins = []
for p in pos.keys():
    embed.append(pos[p].numpy())
    lab.append('sub')
    proteins.append(p)

for p in neg.keys():
    embed.append(neg[p].numpy())
    lab.append('non')
    proteins.append(p)

random.shuffle(lab)
X = pd.DataFrame(embed, index=proteins)
y = pd.Series(lab, index=proteins)

Learn(X, y, "RandomForest")


