import numpy as np 
import sys
from datetime import datetime
import os
from simulation_functions import SimulationSession


def main():

    output_dir = os.path.join('..', '02_Data', 'firing_rates')
    settings_file = 'parameter.yml'
    filename_connectivity = 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median_thresh-0.005.csv'
    filepath_connectivity = os.path.join('..', '02_Data', 'connectivity', filename_connectivity)
    drn_connect_file = 'drn_connectivity_cre-True_hemi-3_grouping-median_thresh-0.005.csv'
    nrAreas = 14
    nr_sessions = 10 # to get some error bars we run several sessions

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # serotonin stimulation strength 
    S_parameters =  [40, 0] 
    # coupling strength G
    G_parameters = [2] #np.arange(0, 4, 1)
    # scaling the tau parameters (to reach desired frequency of UP-DOWN dynamics)
    time_param = 3 

    plot_results = False

    for G in G_parameters:
        G_param = np.round(G, 1)
        print(f'\nSimulation with G = {G_param}')

        for S in S_parameters:
            for session in range(nr_sessions):
                print(f"Run: S={S}, G={G}, session={session}")
                # create the simulation session 
                S_param =  np.round(S, 1)
                sim_session = SimulationSession(output_dir, nrAreas, filepath_connectivity, settings_file, drn_connect_file, G_param, S_param, time_param, session)
                sim_session.start_sim()
                if plot_results:
                    sim_session.plot_results()


if __name__ == '__main__':
   main()