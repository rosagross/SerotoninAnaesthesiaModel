
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

"""
Plot simulation with "usual" serotonin stimulation (targeting E with negative current)
"""

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure4'))
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '02_Data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'atlas_data'))

# load data


# plot down states for one area
