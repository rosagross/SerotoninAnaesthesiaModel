# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style


# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Methods', 'Figure2'))

# load data 
time_constants_states = pd.read_csv(os.path.join(analysed_data_dir, 'state_statistics_local_thetaE-1_beta6_timescales.csv'))

# %% Plot

# x : time constant scaler
# y : state frequency 

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75))

sns.barplot(time_constants_states, x='time_constant', y='state_frequency', ax=axs, errorbar='se', color='grey')

axs.set_title('Time constant determines state frequency')
axs.set_xlabel('Time scale ')
axs.set_ylabel('State frequency')

sns.despine(trim=True, ax=axs)
