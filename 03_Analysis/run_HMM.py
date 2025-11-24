# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from analysis_functions import get_all_states, calc_HMM_states


# %% State analysis global model

output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data', 'firing_rates'))

G_params = [2]
S_params = [0, 40]
beta = 6
thetaE = -1
session_params = pd.DataFrame()
extra = "RateAdj1_B_sessions"
filename_out = ''
sessions = np.arange(0,10,1)
all_analyses = pd.DataFrame()

# %% 

for G in G_params:
    for S in S_params:           
            
        for session in sessions:
            
            print(f"Run: S={S}, G={G}, session={session}")
            # load the file name
            session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}'
            file_dir = os.path.join(output_dir, session_name)
            file_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_{session}'    
    
            # check if any files in the directory end with 'states.csv'
            if any(f.endswith(f'{session}_states.csv') for f in os.listdir(file_dir)):
                pass
            else:
                print('The folder does not contain a file with the ending "states.csv"')
                _ = get_all_states(file_dir, file_name)


# %% State analysis local model

output_dir_local = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data', 'local_circuit_data'))

beta_params = np.arange(0,8,1)
thetaE_params = np.arange(4,13,1)

for thetaE in thetaE_params:

    for beta in beta_params:
        # load the file name
        file_name = f'y_L-1_thetaE-{thetaE}_betaE-{beta}_gI4_gE1'    
        rates = pd.read_csv(os.path.join(output_dir_local, file_name+'.csv'))
        rates = np.array(rates)

        if np.isnan(rates).any():
            print('nan') 
            nan_array = np.empty((10000,))
            nan_array[:] = np.nan
            zhat = nan_array.copy()
            p_down = nan_array.copy()
        else:
            print('no nan')
            zhat, p_down = calc_HMM_states(rates[1,:])

        # safe the states in this dataframe
        states = pd.DataFrame(data={'state': zhat, 'p_down': p_down})
    
        states.to_csv(os.path.join(output_dir_local, file_name+'_states.csv'))


# %% run HMM for different time constants
time_params = [1,2,3,4]
output_dir_local = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data', 'local_circuit_data'))


for time_scale in time_params:

    # load the file name
    file_name = f'y_L-1_thetaE--1_betaE-6_gI4_gE1_timescale{time_scale}'    
    rates = pd.read_csv(os.path.join(output_dir_local, file_name+'.csv'))
    rates = np.array(rates)

    if np.isnan(rates).any():
        print('nan') 
        nan_array = np.empty((10000,))
        nan_array[:] = np.nan
        zhat = nan_array.copy()
        p_down = nan_array.copy()
    else:
        print('no nan')
        zhat, p_down = calc_HMM_states(rates[1,:])

    # safe the states in this dataframe
    states = pd.DataFrame(data={'state': zhat, 'p_down': p_down})

    states.to_csv(os.path.join(output_dir_local, file_name+'_states.csv'))
