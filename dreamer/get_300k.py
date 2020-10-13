from glob import glob
import numpy as np
import os

li = sorted(glob('*.npz'))
n_ep = int(300 * 1000 / 150)
np.array(li[:n_ep])

for file in li[n_ep:]: os.remove(file)