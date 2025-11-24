# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from analysis_functions import compute_brunel, compute_transitions, compute_state_analysis

#import analysis_functions as af

'''
Analyse states: downstate probability, probability of state change
Required: files that ran the HMM analysis for every session to analyse
'''

# directory where state- and firing data is stored
output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data', 'firing_rates'))
output_dir_local = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),  os.pardir, 'ModelData', 'data', 'local_circuit_data'))

# directory for saving analysed data
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, 'ModelData', 'analysed_data'))

# %%

G_params = [2]
S = 40
beta = 6
thetaE = -1
extra = 'RateAdj1_I'
all_sessions = np.arange(0,10,1)
window_duration = 4000
pre_stim = 1000
serotonin_analysis = pd.DataFrame()

# load data for all sessions 
for G in G_params:
    print('G:', G)
    print('S:', S)
    for session in all_sessions:

        # compute down state probability 
        data_df = compute_transitions(output_dir, [G], [S], extra=extra, window_length=window_duration, pre_stim=pre_stim)
        serotonin_analysis = pd.concat((serotonin_analysis, data_df))

serotonin_analysis.to_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G_params}_{extra}_UPDATE.csv'), index=False)

# take the mean for the combined regions 
regions = np.unique(serotonin_analysis['region_name'])
state_transitions = serotonin_analysis.groupby(['region_name', 'S', 'session', 'G', 'time'], as_index=False).mean()

# compute bins
# for this find how many sessions there are in this set and how long one session is 
bin_size = 50
nr_sessions = len(np.unique(state_transitions['session']))
max_time = len(np.unique(state_transitions['time']))
nr_bins = max_time/bin_size
bins = np.linspace(-pre_stim, max_time-pre_stim, int(nr_bins), endpoint=False)

# compute the time points of the bins
digitized = np.digitize(np.unique(state_transitions['time']), bins, right=False)
state_transitions['bin'] = np.tile(digitized, nr_sessions*len(regions)*len([S])*len(G_params))

# average over bins
data_grouped_bins = state_transitions.groupby(['region_name', 'session', 'bin', 'G', 'S'], as_index=False).mean()
data_grouped_bins['time'] = np.tile(bins, nr_sessions*len(regions)*len([S])*len(G_params))/1000

# save averaged time bin data 
data_grouped_bins.to_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S{S}_G{G_params}_{extra}_UPDATE.csv'), index=False)
# %%

# run state analysis for all different option of serotonin targets (targeting either E or I population 
# with an inhibitory or excitatory current)

S = 0
G = [2]
window_duration = 4000
pre_stim = 1000

serotonin_analysis = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1.csv'))
serotonin_analysis_I = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_I.csv'))
serotonin_analysis_J = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_J.csv'))
serotonin_analysis_K = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_analysis_all_regions_S{S}_G{G}_RateAdj1_K.csv'))

regions = np.unique(serotonin_analysis['region_name'])
targets = {'E-' : serotonin_analysis, 'I+' : serotonin_analysis_I, 'I-' : serotonin_analysis_J, 'E+' : serotonin_analysis_K}

# group data into time bins 
bin_size = 200
all_targets_df = pd.DataFrame()

for target_name, target_df in targets.items():
    # 
    # take the mean for the combined regions 
    state_transitions = target_df.groupby(['region_name', 'S', 'session', 'G', 'time'], as_index=False).mean()

    # compute bins
    # for this find how many sessions there are in this set and how long one session is 
    nr_sessions = len(np.unique(state_transitions['session']))
    max_time = len(np.unique(state_transitions['time']))
    nr_bins = max_time/bin_size
    bins = np.linspace(-pre_stim, max_time-pre_stim, int(nr_bins), endpoint=False)

    # compute the time points of the bins and add corresponding target name (e.g., 'E+')
    digitized = np.digitize(np.unique(state_transitions['time']), bins, right=False)
    state_transitions['bin'] = np.tile(digitized, nr_sessions*len(regions)*len([S])*len(G))
    data_grouped_bins = state_transitions.groupby(['region_name', 'session', 'bin', 'G'], as_index=False).mean()
    data_grouped_bins['time'] = np.tile(bins, nr_sessions*len(regions))/1000
    data_grouped_bins['target_name'] = target_name
    
    all_targets_df = pd.concat((all_targets_df, data_grouped_bins))

# save data
all_targets_df.to_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S}_G{G}_UPDATE.csv'))

# %% Run synchrony analysis     

# calculate the states and synchrony quantification (brunel X parameter) for many different G values 
G_parameters = [2]
S = 0
brunel_X_df = pd.DataFrame()
extra = 'RateAdj1_B'
thetaE = -1
beta = 6
sessions = np.arange(0, 10, 1)

for G in G_parameters:

    for ses in sessions:
        G_param = np.round(G, 1)
        session_name = f'14areas_G{float(G)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions'
        file_dir = os.path.join(output_dir, session_name)
        file_name = f'14areas_G{float(G_param)}_S{float(S)}_thetaE{thetaE}_beta{beta}{extra}_sessions_{ses}'
        brunel_X_df = pd.concat((brunel_X_df, compute_brunel(file_dir, file_name)))

brunel_X_df.to_csv(os.path.join(analysed_data_dir, f'brunelX_S{S}_G{G_parameters}{extra}.csv'))

# %% run state analysis (on whole trajectory)

sessions = np.arange(0, 10, 1)
S = 0
G = 2
session_dir = f"14areas_G{float(G)}_S{float(S)}_thetaE-1_beta6RateAdj1_sessions"
states_df = pd.DataFrame()

for ses in sessions:

    file_name = f"14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions_{ses}"
    file_dir = os.path.join(output_dir, session_dir)
    state_stats = compute_state_analysis(file_dir, file_name)
    states_df = pd.concat((states_df, state_stats))

states_df.to_csv(os.path.join(analysed_data_dir, f'totalduration_state_analysis_S{S}_G{G}RateAdj1.csv'))



# %% Local circuit analysis (for phase diagrams): make summary file with state frequency and firing rate info 

state_statistics_local = pd.DataFrame()

beta_params = np.arange(0,8,1)
thetaE_params = np.arange(-5,13,1)
gI_params = [4] # np.arange(0,7,1)
gE_params = [1] # np.arange(0,3.5,0.5)

for thetaE in thetaE_params:
    for beta in beta_params:
        for gI in gI_params:
            for gE in gE_params:
                
                if thetaE >= 4:
                    file_name = f'y_L-1_thetaE-{thetaE}_betaE-{beta}_gI{gI}_gE{gE}'
                else:
                    file_name = f'y_L-1_thetaE-{float(thetaE)}_betaE-{beta}_gI{gI}_gE{gE}'
                f_rates = pd.read_csv(os.path.join(output_dir_local, f'{file_name}.csv'))
                f_rates = np.array(f_rates)
                states = pd.read_csv(os.path.join(output_dir_local, f'{file_name}_states.csv'))
                
                # cast the state time series to list
                time_series = list(states['state'])
                trial_duration = len(time_series)
                
                num_zeros = time_series.count(0)
                num_ones = time_series.count(1)

                # calculate the duration of each state (in number of time steps)
                zero_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 0]
                one_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 1]

                # calculate the frequency of the states (I use the up-state here) 
                f_s = (len(one_durations)/trial_duration)*1000

                # calculate the average state durations 
                d_down = np.mean(zero_durations)
                d_up = np.mean(one_durations)

                # calculate the total duration of each state (in number of time steps)
                total_zero_duration = sum(zero_durations)
                total_one_duration = sum(one_durations)

                # calculate the proportion of time spent in each state
                prop_zero = total_zero_duration / len(time_series)
                prop_one = total_one_duration / len(time_series)
                
                # calculate the mean firing rate for E and I 
                rateE = np.mean(f_rates[0])
                rateI = np.mean(f_rates[1])

                # collect the values in the dataframe 
                state_statistics_local = pd.concat((state_statistics_local, pd.DataFrame(data={'state_frequency': f_s, 'd_down': d_down,
                                                                                'd_up' : d_up, 'p_down' : prop_zero, 'betaE' : beta,
                                                                                'p_up' : prop_one, 'thetaE' : thetaE, 'gE':gE, 'gI':gI,
                                                                                'rateI' : rateI, 'rateE' : rateE, 'time_constant' : time_scale}, index=[0])))

state_statistics_local.to_csv(os.path.join(analysed_data_dir, f'state_statistics_local_thetaE-{thetaE}_betaE-{beta}_gI{gI}_gE{gE}.csv'))

# %% run state analysis for different time constants

thetaE = -1
beta = 6
gI = 4
gE = 1

time_scale_params = [1,2,3,4]
state_statistics_time_scales = pd.DataFrame()

for time_scale in time_scale_params:
                
    file_name = f'y_L-1_thetaE-{thetaE}_betaE-{beta}_gI{gI}_gE{gE}_timescale{time_scale}'
    f_rates = pd.read_csv(os.path.join(output_dir_local, f'{file_name}.csv'))
    f_rates = np.array(f_rates)
    states = pd.read_csv(os.path.join(output_dir_local, f'{file_name}_states.csv'))
    
    # cast the state time series to list
    time_series = list(states['state'])
    trial_duration = len(time_series)
    
    num_zeros = time_series.count(0)
    num_ones = time_series.count(1)

    # calculate the duration of each state (in number of time steps)
    zero_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 0]
    one_durations = [sum(1 for _ in group) for key, group in itertools.groupby(time_series) if key == 1]

    # calculate the frequency of the states (I use the up-state here) 
    f_s = (len(one_durations)/trial_duration)*1000

    # calculate the average state durations 
    d_down = np.mean(zero_durations)
    d_up = np.mean(one_durations)

    # calculate the total duration of each state (in number of time steps)
    total_zero_duration = sum(zero_durations)
    total_one_duration = sum(one_durations)

    # calculate the proportion of time spent in each state
    prop_zero = total_zero_duration / len(time_series)
    prop_one = total_one_duration / len(time_series)
    
    # calculate the mean firing rate for E and I 
    rateE = np.mean(f_rates[0])
    rateI = np.mean(f_rates[1])

    # collect the values in the dataframe 
    state_statistics_time_scales = pd.concat((state_statistics_time_scales, pd.DataFrame(data={'state_frequency': f_s, 'd_down': d_down,
                                                                    'd_up' : d_up, 'p_down' : prop_zero, 'betaE' : beta,
                                                                    'p_up' : prop_one, 'thetaE' : thetaE, 'gE':gE, 'gI':gI,
                                                                    'rateI' : rateI, 'rateE' : rateE, 'time_constant' : time_scale}, index=[0])))

state_statistics_time_scales.to_csv(os.path.join(analysed_data_dir, f'state_statistics_local_thetaE-1_beta6_timescales.csv'))
