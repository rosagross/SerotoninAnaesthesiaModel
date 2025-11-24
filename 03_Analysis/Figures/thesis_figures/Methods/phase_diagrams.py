# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style


'''
1) phase diagram of gE vs gI - to show adjustment of firing rate 
2) phase diagram of thetaE vs beta - to show adjustment of UP/DOWN state frequency
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Methods', 'Figure2'))

# load data 
gIvsgE = pd.read_csv(os.path.join(analysed_data_dir, 'state_statistics_local_thetaE-1_beta6.csv'))
thetaEvsbeta = pd.read_csv(os.path.join(analysed_data_dir, 'state_statistics_local_gI4_gE1.csv'))

# %% gI vs gE phase diagram 

# I and E firing rate
# x axes: values of gE
# y axes: value for gI

color_parameter = 'rateI'
gIvsgE.loc[gIvsgE[color_parameter]>1000, color_parameter] = 0
gIvsgE[color_parameter] = gIvsgE[color_parameter].fillna(0)
gIvsgE[color_parameter+'_log'] = np.log(gIvsgE[color_parameter])

matrix_gIvsgE = gIvsgE.pivot(index='gI', columns='gE', values=color_parameter)

colors, dpi = figure_style()
plt.figure(figsize=(3, 1.75))
ax = sns.heatmap(matrix_gIvsgE, cmap='turbo', square=True, rasterized=True) #, norm=LogNorm(), cbar_kws={'ticks':MaxNLocator(2), 'format':'%.e'})
ax.invert_yaxis()
plt.ylabel('rI slope (gI)')
plt.xlabel('rE slope (gE)')
plt.title('Firing rate population I')
plt.savefig(os.path.join(figure_dir , f"phase_diagram_gIvsgE_{color_parameter}.pdf"), bbox_inches="tight")
plt.show()


# %% thetaE vs beta phase diagram


# state frequency
# x axes: values of thetaE
# y axes: value for beta

thetaEvsbeta.loc[thetaEvsbeta['state_frequency']>15, 'state_frequency'] = np.NaN
thetaEvsbeta = thetaEvsbeta[thetaEvsbeta['betaE']>0] 
matrix_thetaEvsbeta = thetaEvsbeta.pivot(index='betaE', columns='thetaE', values='state_frequency')

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(4, 1.75))
sns.heatmap(matrix_thetaEvsbeta, cmap='turbo', square=True, rasterized=True, ax=axs) #, norm=LogNorm(), cbar_kws={'ticks':MaxNLocator(2), 'format':'%.e'})
axs.invert_yaxis()
plt.ylabel('Adaptation strength (beta)')
plt.xlabel('E threshold (thetaE)')
plt.title('State frequency (%)')
plt.savefig(os.path.join(figure_dir , "phase_diagram_thetaEvsbeta_firingrateI.pdf"), bbox_inches="tight")
plt.show()
