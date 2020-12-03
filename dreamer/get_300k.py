from glob import glob
import numpy as np
import os
import shutil
from tqdm import tqdm

source = 'logdir/mw_push/dreamer_closeup/episodes'
destination = 'logdir/mw_push/dreamer_closeup/offline_300k/episodes'

li = sorted(glob(f'{source}/*.npz'))
k = 300
n_ep = int(k * 1000 / 150)
# np.array(li[:n_ep])
os.makedirs(destination, exist_ok=True)

# Copy files
for file in tqdm(li[:n_ep]):
  name = file.split('/')[-1]
  shutil.copy(file, f'{destination}/{name}')