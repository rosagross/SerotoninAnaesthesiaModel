
#%%
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
REPOPATH = "C:\\Users\\gross\\OneDrive - UvA\\Documents\\MBCS Amsterdam\\Research_Project_2_Champalimaud\\Paper\\SerotoninAnaesthesiaModel"
os.chdir(REPOPATH)
REPOPATH = f"\\\\?\\{os.path.abspath(REPOPATH)}"
# move to that directory to avoid too long paths
# directory where state- and firing data is stored
frate_dir = os.path.join(REPOPATH, '02_Data', 'extended_publication', 'firing_rates')
analysed_data_dir = os.path.join(REPOPATH, '03_Analysis', 'extended_publication', 'analysed_data')
figure_dir = os.path.join(REPOPATH, '03_Analysis', 'Figures')
atlas_dir = os.path.join('03_Analysis', 'atlas_data')


# load data 
file_dir = "14areas_G0.0_S40.0_thetaE-1_beta6_I_sessions"
file_name = "14areas_G0.0_S40.0_thetaE-1_beta6_I_sessions_0"
start = 0*1000
stop = 8*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]

# load state data
states_data = pd.read_csv(os.path.join(frate_dir, file_dir, file_name+'_states.csv'), index_col=0)

# choose area to plot 
region = 12
region_UP = 3

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)
updown_regions = ['Amyg', 'mPFC', 'OLF', 'OFC', 'Pir', 'RSC', 'M2', 'Str', 'Thal', 'Vis']
up_regions = ['Hipp', 'MRN', 'PAG', 'SC']
all_regions = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'Vis']


# plot down states for one area
# %%

colors, dpi = figure_style()
f, axs = plt.subplots(1, 2, figsize=(3, 1.75), dpi=dpi, sharey=True)

# plot rates
axs[0].plot(example_rates[1,:,region], label='I', color='#ff7f0e')
axs[0].plot(example_rates[0,:,region], label='E', color='#1f77b4')

# plot states
data = states_data[states_data['region']==region].reset_index()
data = data.iloc[start:stop]
data['time'] = data['index']/1000
lim = 1.05 * abs(example_rates[1,:,region]).max()

axs[0].imshow(np.expand_dims(data['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[0].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[0].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[0].set_yticks(ticks=[0, 20, 40, 60])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('firing rate (Hz)')
sns.despine(trim=True)
axs[0].set_title(atlas[region][0])


# plot rates UP state region
axs[1].plot(example_rates[1,:,region_UP], label='I', color='#ff7f0e')
axs[1].plot(example_rates[0,:,region_UP], label='E', color='#1f77b4')

data_UP = states_data[states_data['region']==region_UP].reset_index()
data_UP = data_UP.iloc[start:stop]
data_UP['time'] = data_UP['index']/1000

axs[1].imshow(np.expand_dims(data_UP['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[1].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[1].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[1].set_yticks(ticks=[0, 20, 40, 60])
axs[1].set_xlabel('time (s)')
sns.despine(trim=True)
axs[1].set_title(atlas[region_UP][0])