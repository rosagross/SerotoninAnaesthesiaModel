# %%
import numpy as np 
import sys
import os
from datetime import datetime
import os
from simulation_functions import SimulationSession
import matplotlib.pyplot as plt
# %%
REPOPATH = "C:\\Users\\gross\\OneDrive - UvA\\Documents\\MBCS Amsterdam\\Research_Project_2_Champalimaud\\Paper\\SerotoninAnaesthesiaModel"
# move to that directory to avoid too long paths
os.chdir(REPOPATH)

def main():
   
    output_dir = os.path.join('02_Data', 'extended_publication', 'firing_rates')
    settings_file = os.path.join('01_Simulation', 'parameter.yml')
    filename_connectivity = 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median_thresh-0.005.csv'
    filepath_connectivity = os.path.join('02_Data', 'connectivity', filename_connectivity)
    drn_connect_filename = 'drn_connectivity_cre-True_hemi-3_grouping-median_thresh-0.005.csv'
    drn_connect_file = os.path.join('02_Data', 'connectivity', drn_connect_filename)
    nrAreas = 14
    nr_sessions = 1 # to get some error bars we run several sessions

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # serotonin stimulation strength 
    S_parameters =  [0] 
    # coupling strength G
    G_parameters = [0] # np.arange(0, 4, 1)
    # scaling the tau parameters (to reach desired frequency of UP-DOWN dynamics)
    time_param = 3
    run_simulation = False
    plot_results = True
    run_phase_plane_analysis = True
    run_adaptation_nullcline_plots = True

    if run_simulation:
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

    if run_phase_plane_analysis:
        analysis_session = SimulationSession(
            output_dir=output_dir,
            nrAreas=nrAreas,
            filepath_connectivity=filepath_connectivity,
            settings_file=settings_file,
            drn_connect_file=drn_connect_file,
            G=float(G_parameters[0]),
            S=float(S_parameters[0]),
            time_scale=time_param,
            session='phaseplane',
        )
        analysis_session.set_parameter()
        fig, _ = analysis_session.plot_phase_plane_perturbation_comparison(
            area=0,
            external_input=10.0,
            pop_x='E',
            pop_y='I',
            xlim=(-5, 8),
            ylim=(-10, 20),
            grid_size=150,
            coupling_drive=0,
            equilibrium_guess=[12,12,3],
        )
        fig.savefig(os.path.join(REPOPATH, '03_Analysis', 'extended_publication', 'phaseplane_serotonin_stimulation.pdf'))

        if run_adaptation_nullcline_plots:

            fig, _ = analysis_session.plot_nullclines_adaptation_sweep(
                area=0,
                adaptation_values=[0, 1, 2, 3, 4],
                external_input=0.0,
                perturb_mode='I-',
                xlim=(-1, 8),
                ylim=(-5, 20),
                grid_size=80,
            )
            fig.savefig(os.path.join(REPOPATH, '03_Analysis', 'extended_publication', 'phaseplane_adaptation.pdf'))

            g_values = [0,1,2,3]
            fig3, _ = analysis_session.plot_nullclines_coupling_sweep(
                area=1, g_values=g_values,
                external_input=0.0, perturb_mode='I-',
                coupling_drive=1.0,       # constant background from other areas
                xlim=(-1, 8),
                ylim=(-5, 20),
            )
            fig3.savefig(os.path.join(REPOPATH, '03_Analysis', 'extended_publication', 'phaseplane_coupling.pdf'))
            plt.show()

if __name__ == '__main__':
   main()

# %%
