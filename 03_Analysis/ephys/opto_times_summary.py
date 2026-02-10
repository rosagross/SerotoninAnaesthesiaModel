# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:58:00 2026 by Guido Meijer
"""

import numpy as np
from os import path
from glob import glob
import matplotlib.pyplot as plt
from updown_functions import paths

_, data_path = paths()
hz25_paths = glob(path.join(data_path, 'OptoTimes', 'ZFM-05492_2023-02-09*pulse_trains_*.npy'))

all_times = []
for this_path in hz25_paths:
    all_times.append(np.load(this_path))

all_times = np.sort(np.concatenate(all_times))
plt.hist(np.diff(all_times))
    

