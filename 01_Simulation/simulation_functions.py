import os
import os.path as op
import math
import numpy as np 
from datetime import datetime
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import root
#import ctypes

# Enable long paths (optional, but recommended)
#ctypes.windll.kernel32.SetConsoleTitleW(" ", 0)

class SimulationSession():

    def __init__(self, output_dir, nrAreas, filepath_connectivity, settings_file, drn_connect_file, G, S, time_scale, session=''):
        # a session needs parameters and output functions 
        self.nrVars = 3 # E, I and A
        self.nrAreas = nrAreas
        self.initial_cond = np.zeros((self.nrVars, self.nrAreas))
        self.noise_init = np.zeros((2, self.nrAreas))
        self.output_dir = output_dir
        self.output_dir = os.path.abspath(self.output_dir)
        self.output_dir = f"\\\\?\\{self.output_dir}"
        self.filepath_connectivity = filepath_connectivity
        self.settings_file = settings_file
        self.drn_connect_file = drn_connect_file
        self.session = '_' + str(session)
        self.settings = self.load_settings()
        self.time_scale = time_scale
        self.init_parameters(self.settings, G, S)
        self.stimulation_times, self.serotonin_trajectory = self.set_stimulation()
        self.save_settings()

        # generating the random values for the noise 
        self.t_end = self.total_sim * 1000
        self.nrSteps = int((self.t_end-0)/self.dt)
        self.random_vals = np.random.normal(0,1,(2,self.nrSteps,self.nrAreas))


    def load_settings(self):
        
        # load settingsfile 
        with open(self.settings_file, 'r', encoding='utf8') as f_in:
            settings = yaml.safe_load(f_in)

        return settings
    

    def save_settings(self):

        # make a folder where I can save the firing rate together with the setting file and the stimulation times
        extra = "_I_sessions"
        self.file_addon = f'{self.nrAreas}areas_G{self.G}_S{self.S}_thetaE{self.thetaE}_beta{self.betaE}{extra}'
        self.output_dir = op.join(self.output_dir, self.file_addon)

        # if we run several sessions we want to have them all in the same folder and not make a new one with the date and time
        if self.output_dir.endswith('sessions'):
            if not os.path.exists(self.output_dir):
                print('Folder created')
                os.makedirs(self.output_dir)

        else:
            if os.path.exists(self.output_dir):
                print("Warning: output directory already exists. Renaming to avoid overwriting.")
                self.output_dir = self.output_dir + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
                os.makedirs(self.output_dir)
            else:
                print('Folder created')
                os.makedirs(self.output_dir)
        

        settings_out = op.join(self.output_dir, self.file_addon + self.session + '_expsettings.yml')
        
        with open(settings_out, 'w') as f_out:  # write settings to disk
            yaml.dump(self.settings, f_out, indent=4, default_flow_style=False)

        # safe the stimulation array in a csv
        outdir = self.output_dir
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outname = self.file_addon + self.session + '_stimtimes.csv'
        outpath_full = os.path.join(outdir, outname)
        stim_times = pd.DataFrame(self.stimulation_times)
        stim_times.to_csv(outpath_full, index=False)

    
    def init_parameters(self, config, G, S):
        # connectivity parameters
        self.Jee = config['Parameter']['Jee']
        self.Jei = config['Parameter']['Jei']
        self.Jie = config['Parameter']['Jie']
        self.Jii = config['Parameter']['Jii']

        # effective threshold for population activation
        self.thetaE = config['Parameter']['thetaE']
        self.thetaI = config['Parameter']['thetaI']
        self.betaE = config['Parameter']['betaE']
        self.betaI = config['Parameter']['betaI']
        self.Eslope = config['Parameter']['Eslope']
        self.Edesp = config['Parameter']['Edesp']
        self.Islope = config['Parameter']['Islope']
        self.Idesp = config['Parameter']['Idesp']

        # thetaE and betaE for Up-state regions
        self.thetaE_UP = config['Parameter']['thetaE_UP']
        self.betaE_UP = config['Parameter']['betaE_UP']
        self.Up_areas = config['Parameter']['Up_areas']

        # thetaE and betaE for unclear Up/Down-state regions
        self.thetaE_un = config['Parameter']['thetaE_un']
        self.betaE_un = config['Parameter']['betaE_un']
        self.un_areas = config['Parameter']['un_areas']

        # G, S, and stim_ITI parameters
        if G is not None:
            config['Parameter']['G'] = float(G)

        self.G = config['Parameter']['G']

        if S is not None:
            config['Parameter']['S'] = float(S)

        self.S = config['Parameter']['S']
        self.stim_ITI = config['Parameter']['stim_ITI']
        self.stim_dur = config['Parameter']['stim_dur']
        self.post_stim_time = config['Parameter']['post_stim_time']

        # time constants and noise parameter
        self.tauE = config['Parameter']['tauE'] * self.time_scale
        self.tauI = config['Parameter']['tauI'] * self.time_scale
        self.tauAdapt = config['Parameter']['tauAdapt'] * self.time_scale
        self.tauN = config['Parameter']['tauN'] * self.time_scale
        self.sigmaN = config['Parameter']['sigmaN'] 

        # session settings
        self.total_sim = config['Session settings']['total_sim']
        self.sigmaN = config['Session settings']['sigmaN']
        self.dt = config['Session settings']['dt']
        self.x0 = config['Session settings']['x0']


    def set_stimulation(self):
        '''
        Compute the array with serotonin stimulation times. 
        '''

        # check if the total simulation window is big enough for at least one stimulation
        if self.stim_ITI[1] + 4 > self.total_sim:
            print('! Warning: Choose an ITI that fits within the stimulation window or increase the simulation time !')

        # create a time window of 1 sec stimulation time 
        total = self.total_sim * 1000
        stimulation_duration = self.stim_dur * 1000
        end = 0
        post_stim_time = self.post_stim_time * 1000 # serotonin is only done after 2 second post-stimulation
        stimulation = True 
        stimulation_array = []

        while stimulation:
            pause = np.random.randint(self.stim_ITI[0]*1000, self.stim_ITI[1]*1000)
            start = end + pause
            end = start + stimulation_duration
            stimulation_array.append([start, (end + post_stim_time - 1)])

            # we need some space after the stimulation period to observe if there is a rebound up state
            if end + 4000 > total:
                stimulation = False
                stimulation_array.pop()

        # calculate the time trajectory of serotonin stimulation (curve describing its effect over time)  
        start_time = 0  # Start time in seconds
        end_time = 3  # End time in seconds
        step_size = 0.0002  # Step size in seconds
        time = np.arange(start_time, end_time, step_size)
        trajectory = np.array([0, 1, 0.7, 0.5, 0.45, 0.3, 0])
        # interpolate the trajectory values at the specified time points
        interpolated_trajectory = np.interp(time, np.linspace(start_time, end_time, len(trajectory)), trajectory)

        return np.array(stimulation_array), interpolated_trajectory


    def start_sim(self):
        
        # setup connectivity matrix and the parameter arrays
        self.set_connectivity(self.drn_connect_file)
        self.set_parameter()

        # let the integration happen!
        self.output_y, self.output_noise = self.integrator_RK4()
        self.output_y = np.moveaxis(self.output_y, 0, -2)
        self.save_output()


    def set_connectivity(self, drn_connect_file):
        '''
        In this function the connectivity matrix is setup 
        '''

        # connectivity matrix for the 14 regions
        self.c_matrix = pd.read_csv(self.filepath_connectivity)
        self.c_matrix.drop('Unnamed: 0', inplace=True, axis=1)
        self.c_matrix = np.array(self.c_matrix)

        # uncomment this if you want a uniform connectivity matrix
        #self.c_matrix = np.ones((self.nrAreas,self.nrAreas))*mean_connect

        np.fill_diagonal(self.c_matrix, 0)

        # connectivity of DRN 
        self.drn_connect = pd.read_csv(drn_connect_file)


    def set_parameter(self):
        ''' 
        Set up the thetaE and betaE parameter. We take the values from the parameter/main file
        and make a matrix out of it so that we can have the correct parameter combination
        for every area. Some areas show up and down states, others only show upstates.
        '''    

        # this is replicated from the original C++ code  
        # the thetaE value entered in the parameter file is the one that can be used in  
        # figure 5 of the Jercog (2017) paper 
        thetaE_thresh = self.Edesp - self.thetaE
        thetaE_Up_thresh = self.Edesp - self.thetaE_UP
        thetaE_un_thresh = self.Edesp - self.thetaE_un

        self.thetaE_array = np.empty(self.nrAreas)
        self.thetaE_array.fill(thetaE_thresh)
        self.betaE_array = np.empty(self.nrAreas)
        self.betaE_array.fill(self.betaE)

        # for areas with UP-states
        self.thetaE_array[self.Up_areas] = thetaE_Up_thresh
        self.betaE_array[self.Up_areas] = self.betaE_UP

        # for areas with unclear UP/DOWN states
        self.thetaE_array[self.un_areas] = thetaE_un_thresh
        self.betaE_array[self.un_areas] = self.betaE_un


    def derivatives(self, y, n):
        '''
        Parameter:
        y : array, 3 x nrAreas time series of the firing rate values of E, I and A. 
        n : array, 3 x nrAreas time series of noise (Ornstein-Uhlenbeck).
        returns : dy, 3 x nrAreas element array with the derivatives of E, I and A
        '''

        dy = np.zeros((3,self.nrAreas))

        # derivative of E - rate
        # aux is a dummy variable for part of the derivative
        #print(self.I)
        aux = self.Jee*y[0] - self.Jei*y[1] + self.thetaE_array + n[0] - y[2] + self.G * np.matmul(self.c_matrix, y[0]) #- self.I

        for area in np.arange(self.nrAreas):
            if aux[area] <= self.Edesp:
                dy[0][area] = -y[0][area]/self.tauE
            else:
                dy[0][area] = (-y[0][area] + ((aux[area]-self.Edesp)*self.Eslope[area]))/self.tauE
        
        # derivative of I - rate 
        aux = self.Jie*y[0] - self.Jii*y[1] + self.thetaI + n[1] + self.I
        for area in np.arange(self.nrAreas):
            if aux[area] <= self.Idesp:
                dy[1][area] = -y[1][area]/self.tauI
            else:
                dy[1][area] = (-y[1][area] + self.Islope[area]*(aux[area]-self.Idesp))/self.tauI

        # derivative of A - adaptation rate
        dy[2] = (-y[2]+self.betaE_array*y[0])/self.tauAdapt

        return dy

    def _get_pop_index(self, pop):
        mapping = {'E': 0, 'I': 1, 'A': 2}
        if isinstance(pop, str):
            pop = pop.upper()
            if pop not in mapping:
                raise ValueError("Population must be one of: 'E', 'I', 'A'.")
            return mapping[pop]
        pop = int(pop)
        if pop not in (0, 1, 2):
            raise ValueError('Population index must be 0, 1, or 2.')
        return pop

    def _validate_area(self, area):
        area = int(area)
        if area < 0 or area >= self.nrAreas:
            raise ValueError(f'Area index {area} out of bounds for {self.nrAreas} areas.')
        return area

    def local_dynamics(self, state, area, external_input=0.0, perturb_mode='I+', coupling_drive=0.0,
                       noise=None):
        """
        Deterministic local dynamics for one area.
        """
        self.set_parameter()

        area = self._validate_area(area)
        state = np.asarray(state, dtype=float).reshape(3,)
        if noise is None:
            noise = np.zeros(2)
        noise = np.asarray(noise, dtype=float).reshape(2,)

        e_val, i_val, a_val = state
        theta_e = self.thetaE_array[area]
        beta_e = self.betaE_array[area]
        #print('beta', beta_e, 'area', area, perturb_mode)
        ext_i = float(external_input) # always positive, the mode determines target

        aux_e = (
            self.Jee * e_val
            - self.Jei * i_val
            + theta_e
            + noise[0]
            - a_val
            + self.G * (coupling_drive * e_val)
        ) 
        if perturb_mode=='I-':
            aux_e = aux_e - ext_i

        if aux_e <= self.Edesp:
            d_e = -e_val / self.tauE
        else:
            d_e = (-e_val + ((aux_e - self.Edesp) * self.Eslope[area])) / self.tauE
        
        aux_i = self.Jie * e_val - self.Jii * i_val + self.thetaI + noise[1]
        if perturb_mode=='I+':
            aux_i = aux_i + ext_i

        if aux_i <= self.Idesp:
            d_i = -i_val / self.tauI
        else:
            d_i = (-i_val + self.Islope[area] * (aux_i - self.Idesp)) / self.tauI

        d_a = (-a_val + beta_e * e_val) / self.tauAdapt
        return np.array([d_e, d_i, d_a], dtype=float)

    def find_local_equilibrium(self, area, external_input=0.0, perturb_mode='I+', initial_guess=None,
                               coupling_drive=0.0, noise=None, tol=1e-10):
        """
        Solve for one local equilibrium (E, I, A) in a selected area.
        """
        self.set_parameter()
        area = self._validate_area(area)
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0, 0.0], dtype=float)
        initial_guess = np.asarray(initial_guess, dtype=float).reshape(3,)

        sol = root(
            lambda x: self.local_dynamics(
                x,
                area=area,
                external_input=external_input,
                perturb_mode=perturb_mode,
                coupling_drive=coupling_drive,
                noise=noise,
            ),
            x0=initial_guess,
            tol=tol,
        )

        residual = self.local_dynamics(
            sol.x,
            area=area,
            external_input=external_input,
            perturb_mode=perturb_mode,
            coupling_drive=coupling_drive,
            noise=noise,
        )
        return {
            'equilibrium': np.asarray(sol.x, dtype=float),
            'success': bool(sol.success),
            'message': str(sol.message),
            'residual_norm': float(np.linalg.norm(residual)),
            'nfev': int(getattr(sol, 'nfev', -1)),
            'initial_guess': initial_guess,
        }

    def find_local_equilibria(self, area, external_input=0.0, perturb_mode='I+', initial_guesses=None,
                              coupling_drive=0.0, noise=None, tol=1e-10,
                              dedupe_tol=1e-3):
        """
        Multi-start equilibrium search for one area.
        """
        if initial_guesses is None:
            initial_guesses = [
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [40.0, 30.0, 20.0],
                [10.0, 30.0, 60.0],
            ]

        equilibria = []
        for guess in initial_guesses:
            result = self.find_local_equilibrium(
                area=area,
                external_input=external_input,
                perturb_mode=perturb_mode,
                initial_guess=guess,
                coupling_drive=coupling_drive,
                noise=noise,
                tol=tol,
            )
            if not result['success'] or result['residual_norm'] > 1e-6:
                continue
            candidate = result['equilibrium']
            is_new = True
            for prev in equilibria:
                if np.linalg.norm(candidate - prev['equilibrium']) < dedupe_tol:
                    is_new = False
                    break
            if is_new:
                equilibria.append(result)
        return equilibria

    def jacobian_at_equilibrium(self, equilibrium, area, external_input=0.0, perturb_mode='I+',
                                coupling_drive=0.0, noise=None):
        """
        Branch-aware Jacobian matrix at one equilibrium point.
        """
        self.set_parameter()
        area = self._validate_area(area)
        equilibrium = np.asarray(equilibrium, dtype=float).reshape(3,)
        if noise is None:
            noise = np.zeros(2)
        noise = np.asarray(noise, dtype=float).reshape(2,)

        e_val, i_val, a_val = equilibrium
        theta_e = self.thetaE_array[area]
        ext_i = float(external_input)

        aux_e = (
            self.Jee * e_val
            - self.Jei * i_val
            + theta_e
            + noise[0]
            - a_val
            + self.G * (coupling_drive * e_val)
        )
        if perturb_mode=='I-':
            aux_e = aux_e - ext_i

        aux_i = self.Jie * e_val - self.Jii * i_val + self.thetaI + noise[1] 
        if perturb_mode=='I+':
            aux_i = aux_i + ext_i

        dphi_e = 0.0 if aux_e <= self.Edesp else float(self.Eslope[area])
        dphi_i = 0.0 if aux_i <= self.Idesp else float(self.Islope[area])


        jac = np.array([
            [(-1.0 + dphi_e * (self.Jee + self.G)) / self.tauE, (-dphi_e * self.Jei) / self.tauE, -1.0 / self.tauE],
            [(dphi_i * self.Jie) / self.tauI, (-1.0 - dphi_i * self.Jii) / self.tauI, 0.0],
            [self.betaE_array[area] / self.tauAdapt, 0.0, -1.0 / self.tauAdapt],
        ], dtype=float)
        return jac

    def eigenvalues_at_equilibrium(self, equilibrium, area, external_input=0.0, perturb_mode='I+',
                                   coupling_drive=0.0, noise=None):
        jac = self.jacobian_at_equilibrium(
            equilibrium=equilibrium,
            area=area,
            external_input=external_input,
            perturb_mode=perturb_mode,
            coupling_drive=coupling_drive,
            noise=noise,
        )
        eigvals = np.linalg.eigvals(jac)
        real_parts = np.real(eigvals)
        if np.all(real_parts < -1e-9):
            stability = 'stable'
        elif np.any(real_parts > 1e-9) and np.any(real_parts < -1e-9):
            stability = 'saddle'
        elif np.any(real_parts > 1e-9):
            stability = 'unstable'
        else:
            stability = 'marginal'
        return {
            'jacobian': jac,
            'eigenvalues': eigvals,
            'stability': stability,
        }

    def plot_phase_plane(self, area, external_input=0.0, perturb_mode='I+', pop_x='E', pop_y='I',
                         equilibrium=None, fixed_values=None, coupling_drive=0.0,
                         noise=None, xlim=(0.0, 40.0), ylim=(0.0, 40.0), grid_size=45, trajectories=None,
                         ax=None, title=None):
        """
        Plot a 2D phase plane for a local area with fixed third variable(s).
        """
        self.set_parameter()
        area = self._validate_area(area)
        idx_x = self._get_pop_index(pop_x)
        idx_y = self._get_pop_index(pop_y)
        if idx_x == idx_y:
            raise ValueError('pop_x and pop_y must be different populations.')

        if equilibrium is None:
            print('Pass equilibrium!')

        equilibrium = np.asarray(equilibrium, dtype=float).reshape(3,)

        fixed = equilibrium.copy()

        if fixed_values is not None:
            for pop_name, val in fixed_values.items():
                fixed[self._get_pop_index(pop_name)] = float(val)

        def _slice_nullcline_system(xy):
            state = fixed.copy()
            state[idx_x] = xy[0]
            state[idx_y] = xy[1]
            d_state = self.local_dynamics(
                state,
                area=area,
                external_input=external_input,
                perturb_mode=perturb_mode,
                coupling_drive=coupling_drive,
                noise=noise,
            )
            return np.array([d_state[idx_x], d_state[idx_y]], dtype=float)

        slice_guess = np.array([equilibrium[idx_x], equilibrium[idx_y]], dtype=float)
        slice_sol = root(_slice_nullcline_system, x0=slice_guess)
        marker_state = fixed.copy()
        if slice_sol.success and np.all(np.isfinite(slice_sol.x)):
            marker_state[idx_x] = float(slice_sol.x[0])
            marker_state[idx_y] = float(slice_sol.x[1])
        else:
            marker_state[idx_x] = float(equilibrium[idx_x])
            marker_state[idx_y] = float(equilibrium[idx_y])

        x_vals = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_size))
        y_vals = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_size))
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        dx = np.zeros_like(x_grid, dtype=float)
        dy = np.zeros_like(y_grid, dtype=float)

        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                state = fixed.copy()
                state[idx_x] = x_grid[i, j]
                state[idx_y] = y_grid[i, j]
                d_state = self.local_dynamics(
                    state,
                    area=area,
                    external_input=external_input,
                    perturb_mode=perturb_mode,
                    coupling_drive=coupling_drive,
                    noise=noise,
                )
                dx[i, j] = d_state[idx_x]
                dy[i, j] = d_state[idx_y]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4.5, 4))

        speed = np.sqrt(dx**2 + dy**2)
        ax.streamplot(x_grid, y_grid, dx, dy, color=speed, cmap='viridis', density=1.1, linewidth=0.8)
        ax.contour(x_grid, y_grid, dx, levels=[0], colors='tab:blue', linewidths=2)
        ax.contour(x_grid, y_grid, dy, levels=[0], colors='tab:red', linewidths=2)

        eq_x = marker_state[idx_x]
        eq_y = marker_state[idx_y]
        eig_info = self.eigenvalues_at_equilibrium(
            equilibrium=marker_state,
            area=area,
            external_input=external_input,
            perturb_mode=perturb_mode,
            coupling_drive=coupling_drive,
            noise=noise,
        )
        ax.scatter([eq_x], [eq_y], s=35, color='red', zorder=5)
        ax.annotate(
            eig_info['stability'],
            (eq_x, eq_y),
            textcoords='offset points',
            xytext=(6, 4),
            fontsize=8,
            color='red',
        )

        if trajectories:
            for init in trajectories:
                state = np.asarray(init, dtype=float).reshape(3,)
                traj = [state.copy()]
                dt = 0.5
                for _ in range(250):
                    d_state = self.local_dynamics(
                        state,
                        area=area,
                        external_input=external_input,
                        perturb_mode=perturb_mode,
                        coupling_drive=coupling_drive,
                        noise=noise,
                    )
                    state = state + dt * d_state
                    traj.append(state.copy())
                traj = np.asarray(traj)
                ax.plot(traj[:, idx_x], traj[:, idx_y], color='k', lw=0.8, alpha=0.8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("E Firing Rate (Hz)")
        ax.set_ylabel("I Firing Rate (Hz)")
        if title is None:
            title = f'Area {area} ({perturb_mode}, ext={external_input:g})'
        ax.set_title(title)
        return ax

    def find_local_equilibria_grid(self, area, external_input=0.0, perturb_mode='I+',
                                e_range=(0, 40), i_range=(0, 40), n=6, dedupe_tol=1e-2,
                                coupling_drive=0.0, noise=None):
        """Grid multi-start equilibrium search."""
        self.set_parameter()
        e_starts = np.linspace(e_range[0], e_range[1], n)
        i_starts = np.linspace(i_range[0], i_range[1], n)
        
        equilibria = []
        for e0 in e_starts:
            for i0 in i_starts:
                # A at steady state given E: A* = betaE * E
                a0 = float(self.betaE_array[area]) * e0
                guess = np.array([e0, i0, a0])
                result = self.find_local_equilibrium(
                    area=area, external_input=external_input,
                    perturb_mode=perturb_mode, initial_guess=guess,
                    coupling_drive=coupling_drive,
                    noise=noise,
                )
                if not result['success'] or result['residual_norm'] > 1e-4:
                    continue
                if np.any(result['equilibrium'] < -0.1):  # reject unphysical
                    continue
                candidate = result['equilibrium']
                is_new = all(
                    np.linalg.norm(candidate - p['equilibrium']) > dedupe_tol
                    for p in equilibria
                )
                if is_new:
                    equilibria.append(result)

        return equilibria
    
    def debug_equilibria(self, area, external_input=0.0, perturb_mode='I+', n=6):
        """Print all raw solver results before filtering."""
        self.set_parameter()

        slopeE = float(self.Eslope[area])
        slopeI = float(self.Islope[area])
        betaE  = float(self.betaE_array[area])
        thetaE = float(self.thetaE_array[area])

        e_vals = np.linspace(0, 20, 2000)
        f_vals = np.zeros_like(e_vals)

        for k, e in enumerate(e_vals):
            a_ss    = betaE * e
            drive_i = self.Jie * e + self.thetaI - self.Idesp
            if perturb_mode == 'I+':
                drive_i += external_input
            i_ss = 0.0 if drive_i <= 0 else slopeI * drive_i / (1.0 + slopeI * self.Jii)

            drive_e = self.Jee * e - self.Jei * i_ss + thetaE - a_ss - self.Edesp
            if perturb_mode == 'I-':
                drive_e -= external_input
            f_vals[k] = -e if drive_e <= 0 else -e + slopeE * drive_e

        plt.figure(figsize=(6, 3))
        plt.plot(e_vals, f_vals, 'k', lw=1.5)
        plt.axhline(0, color='r', lw=0.8, ls='--')
        plt.xlabel('E');  plt.ylabel('dE/dt (tauE scaled)')
        plt.title(f'1D E residual — zero crossings = equilibria (mode={perturb_mode}, ext={external_input})')
        plt.tight_layout()
        plt.show()

        e_starts = np.linspace(0, 40, n)
        i_starts = np.linspace(0, 40, n)

        for e0 in e_starts:
            for i0 in i_starts:
                a0 = float(self.betaE_array[area]) * e0
                guess = np.array([e0, i0, a0])
                result = self.find_local_equilibrium(
                    area=area, external_input=external_input,
                    perturb_mode=perturb_mode, initial_guess=guess,
                )
                eq = result['equilibrium']
                res = result['residual_norm']
                # print everything, no filtering
                print(f"guess=({e0:.1f},{i0:.1f},{a0:.1f}) -> "
                    f"eq=({eq[0]:.3f},{eq[1]:.3f},{eq[2]:.3f}) "
                    f"residual={res:.2e} success={result['success']}")

    def plot_phase_plane_perturbation_comparison(self, area, external_input=20.0, pop_x='E', pop_y='I',
                                             equilibrium_guess=None, fixed_values=None, coupling_drive=0.0,
                                             noise=None, xlim=(0.0, 40.0),
                                             ylim=(0.0, 40.0), grid_size=45, trajectories=None):
        self.set_parameter()
        area = self._validate_area(area)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        stability_colors = {'stable': 'green', 'saddle': 'orange',
                            'unstable': 'red', 'marginal': 'grey'}

        for ax, mode in zip(axs, ('I+', 'I-')):

            # first find equilibria with no fixed A to locate the up-state
            equilibria = self.find_local_equilibria_grid(
                external_input=external_input,
                area=area,
                perturb_mode=mode,
                e_range=(0, 40),
                i_range=(0, 40),
                n=6,
                dedupe_tol=1e-2,
                coupling_drive=coupling_drive,
                noise=noise,
            )

            # pick the highest-E equilibrium as the up-state for this mode
            if equilibria:
                e_upstate = max(eq['equilibrium'][0] for eq in equilibria)
            else:
                e_upstate = (xlim[0] + xlim[1]) / 2.0

            # self-consistent A for this mode's up-state
            fixed_adaptation = float(self.betaE_array[area]) * e_upstate

            # build slice fixed values
            slice_fixed_values = {'A': fixed_adaptation}
            if fixed_values is not None:
                slice_fixed_values.update(fixed_values)

            # filter equilibria to those near this A slice
            equilibria_at_slice = [
                eq for eq in equilibria
                if abs(eq['equilibrium'][2] - fixed_adaptation) < 2.0
            ]

            ref_eq = (equilibria_at_slice[0]['equilibrium'] if equilibria_at_slice
                    else np.array([e_upstate, e_upstate, fixed_adaptation]))

            self.plot_phase_plane(
                area=area,
                external_input=external_input,
                perturb_mode=mode,
                pop_x=pop_x,
                pop_y=pop_y,
                equilibrium=ref_eq,
                fixed_values=slice_fixed_values,
                coupling_drive=coupling_drive,
                noise=noise,
                xlim=xlim,
                ylim=ylim,
                grid_size=grid_size,
                trajectories=trajectories,
                ax=ax,
                title=f'{mode} (ext={external_input:g}, A={fixed_adaptation:.2f})',
            )

            for eq in equilibria:
                eig_info = self.eigenvalues_at_equilibrium(
                    equilibrium=eq['equilibrium'],
                    area=area,
                    external_input=external_input,
                    perturb_mode=mode,
                )
                stab = eig_info['stability']
                ex = eq['equilibrium'][self._get_pop_index(pop_x)]
                ey = eq['equilibrium'][self._get_pop_index(pop_y)]
                ax.scatter([ex], [ey], s=60, color=stability_colors.get(stab, 'red'),
                        zorder=6, edgecolors='k', linewidths=0.5)
                ax.annotate(stab, (ex, ey), textcoords='offset points',
                            xytext=(6, 4), fontsize=8,
                            color=stability_colors.get(stab, 'red'))

        fig.suptitle(f'Phase-plane comparison (area={area}, populations {pop_x}-{pop_y})')
        fig.tight_layout()
        return fig, axs

    def plot_nullclines_adaptation_sweep(self, area, adaptation_values, external_input=0.0, perturb_mode='I+',
                                         coupling_drive=0.0, noise=None,
                                         xlim=(0.0, 45.0), ylim=(0.0, 45.0), grid_size=80):
        """
        Plot E and I nullclines for multiple fixed adaptation (A) values.
        """
        self.set_parameter()
        area = self._validate_area(area)
        adaptation_values = np.asarray(adaptation_values, dtype=float).reshape(-1)
        if adaptation_values.size == 0:
            raise ValueError('adaptation_values must contain at least one value.')

        e_vals = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_size))
        i_vals = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_size))
        e_grid, i_grid = np.meshgrid(e_vals, i_vals)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        colors_e = plt.cm.Blues(np.linspace(0.45, 0.95, adaptation_values.size))
        colors_i = plt.cm.Oranges(np.linspace(0.45, 0.95, adaptation_values.size))

        for idx, a_val in enumerate(adaptation_values):
            d_e_grid = np.zeros_like(e_grid, dtype=float)
            d_i_grid = np.zeros_like(i_grid, dtype=float)

            for i in range(e_grid.shape[0]):
                for j in range(e_grid.shape[1]):
                    state = np.array([e_grid[i, j], i_grid[i, j], a_val], dtype=float)
                    d_state = self.local_dynamics(
                        state=state,
                        area=area,
                        external_input=external_input,
                        perturb_mode=perturb_mode,
                        coupling_drive=coupling_drive,
                        noise=noise,
                    )
                    d_e_grid[i, j] = d_state[0]
                    d_i_grid[i, j] = d_state[1]

            axs[0].contour(e_grid, i_grid, d_e_grid, levels=[0], colors=[colors_e[idx]], linewidths=1.6)
            axs[1].contour(e_grid, i_grid, d_i_grid, levels=[0], colors=[colors_i[idx]], linewidths=1.6)
            axs[0].contour(e_grid, i_grid, d_i_grid, levels=[0], colors='red', linewidths=1.6, linestyles='dashed')
            axs[0].plot([], [], color=colors_e[idx], lw=2, label=f'A={a_val:g}')
            axs[1].plot([], [], color='red', lw=2, label=f'A={a_val:g}')
            axs[0].plot([], [], color='red', lw=2)

        axs[0].set_title('E and I nullclines')
        axs[1].set_title('I nullclines')
        for ax in axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
            ax.set_xlabel('E Firing Rate (Hz)')
            ax.set_ylabel('I Firing Rate (Hz)')
            ax.legend(title='Adaptation', fontsize=8, title_fontsize=8)

        fig.suptitle(f'Nullclines across adaptation values (area={area}, mode={perturb_mode}, ext={external_input:g})')
        fig.tight_layout()
        return fig, axs


    def plot_nullclines_coupling_sweep(self, area, g_values, external_input=0.0,
                                    perturb_mode='I+', coupling_drive=0.0,
                                    xlim=(0.0, 40.0),
                                    ylim=(0.0, 40.0), nc_size=300):
        """
        Plot E- and I-nullclines for a sweep of G (global coupling) values.

        For each G value the E-nullcline shifts because G enters aux_e via:
            G * (coupling_drive * E)
        The I-nullcline is unaffected by G (it does not appear in aux_i).

        Parameters
        ----------
        area            : int, area index
        g_values        : iterable of G values to sweep
        external_input  : float, perturbation strength
        perturb_mode    : 'I+' or 'I-'
        coupling_drive  : float, constant background network drive
        xlim, ylim      : axis limits
        nc_size         : int, grid resolution for nullcline smoothness
        """
        self.set_parameter()
        area = self._validate_area(area)
        g_values = list(g_values)

        e_vals = np.linspace(float(xlim[0]), float(xlim[1]), nc_size)
        i_vals = np.linspace(float(ylim[0]), float(ylim[1]), nc_size)
        e_grid, i_grid = np.meshgrid(e_vals, i_vals)

        colors = plt.cm.Blues(np.linspace(0.1, 0.9, len(g_values)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        ax_e, ax_i = axes

        # find up-state E at G=0 to anchor the adaptation slice
        g_orig = self.G
        self.G = 0.0
        equilibria = self.find_local_equilibria_grid(
            area=area, external_input=external_input,
            perturb_mode=perturb_mode, e_range=xlim, i_range=ylim, n=8,
        )
        self.G = g_orig

        if equilibria:
            e_upstate = max(eq['equilibrium'][0] for eq in equilibria)
        else:
            e_upstate = (xlim[0] + xlim[1]) / 2.0
        fixed_adaptation = float(self.betaE_array[area]) * e_upstate

        for color, g in zip(colors, g_values):
            self.G = g  # temporarily set G for local_dynamics calls

            dx_grid = np.zeros_like(e_grid)
            dy_grid = np.zeros_like(i_grid)

            for i in range(nc_size):
                for j in range(nc_size):
                    state = np.array([e_grid[i, j], i_grid[i, j], fixed_adaptation])
                    d_state = self.local_dynamics(
                        state=state,
                        area=area,
                        external_input=external_input,
                        perturb_mode=perturb_mode,
                        coupling_drive=coupling_drive,
                    )
                    dx_grid[i, j] = d_state[0]
                    dy_grid[i, j] = d_state[1]

            # E-nullcline (dE/dt = 0) — shifts with G
            ax_e.contour(e_grid, i_grid, dx_grid, levels=[0],
                        colors=[color], linewidths=1.8)
            # I-nullcline (dI/dt = 0) — does not shift with G, but plot for reference
            ax_i.contour(e_grid, i_grid, dy_grid, levels=[0],
                        colors='red', linewidths=1.8)
            # overlay I-nullcline on E-nullcline panel too
            ax_e.contour(e_grid, i_grid, dy_grid, levels=[0],
                        colors='red', linewidths=1.8, linestyles='dashed')

            # dummy line for legend
            ax_e.plot([], [], color=color, lw=2, label=f'G={g:.2g}')
            ax_i.plot([], [], color=color, lw=2, label=f'G={g:.2g}')

        self.G = g_orig  # restore original G

        ax_e.set_title('E-nullcline (solid) + I-nullcline (dashed)')
        ax_i.set_title('I-nullcline')
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
            ax.set_xlabel('E Firing Rate (Hz)')
            ax.set_ylabel('I Firing Rate (Hz)')
            ax.legend(title='G', fontsize=8, title_fontsize=8)

        fig.suptitle(
            f'Nullclines across G values — area {area}, A={fixed_adaptation:.2f}, '
            f'drive={coupling_drive}'
        )
        fig.tight_layout()
        return fig, axes


    def integrator_RK4(self):
        '''
        The firing rate is computed analytically by using the Runge-Kutta method.  
        '''

        noiseDummy1 = np.exp(-self.dt/self.tauN)
        noiseDummy2 = math.sqrt(((2*(self.sigmaN**2)/self.tauN)*self.tauN*0.5)*(1-math.exp(-self.dt/self.tauN)**2))
        rk4Aux1=self.dt*0.500000000 # (1/2)
        rk4Aux2=self.dt*0.166666666 # (1/6)
        Tdt = int(1/self.dt)
        tsteps = np.arange(self.dt, self.t_end, self.dt)

        y_current = self.initial_cond # keeps track of the current value for the rate 
        noise_current = self.noise_init
        noise = [] 
        noise.append(noise_current)
        # time series per population/variable
        y = []
        y.append(y_current)

        # time counter for when to save the computed value
        k = 0 
        # intitial serotonin stimulation is 0
        self.I = 0

        # stimulation time countdown (to index the correct stimulation value over time)
        
        stim_countdown = 0 #  np.arange(int((self.stim_dur+self.post_stim_time)*1000/self.dt))

        # for every time step, we now have to find the solution of the derivative by integrating 
        for iter, step in enumerate(tsteps):

            # calculate the derivatives
            aux1 = self.derivatives(y_current, noise_current)
            aux2 = y_current + rk4Aux1 * aux1
    
            aux3 = self.derivatives(aux2, noise_current)
            aux2 = y_current + rk4Aux1 * aux3

            aux4 = self.derivatives(aux2, noise_current)
            aux2 = y_current + self.dt * aux4

            aux4 += aux3

            aux3 = self.derivatives(aux2, noise_current)

            y_current = y_current + rk4Aux2 * (aux1+aux3 + 2*aux4)

            # noise for every area individually
            noise_current[0] = noise_current[0] * noiseDummy1+noiseDummy2*self.random_vals[0,iter,:]
            noise_current[1] = noise_current[1] * noiseDummy1+noiseDummy2*self.random_vals[1,iter,:]
            
            # START of serotonin stimulation   
            if (round(step, 1) in self.stimulation_times[:,0]) or (stim_countdown > 0):
                self.I = np.squeeze(self.drn_connect * self.S * self.serotonin_trajectory[stim_countdown])
                stim_countdown += 1 

            # END of serotonin stimulation
            if round(step, 1) in self.stimulation_times[:,1]:
                self.I = 0
                stim_countdown = 0

            k+= 1
            if k == Tdt:
                y.append(y_current)
                noise.append(noise_current)
                k = 0 
                
        return np.array(y), np.array(noise).T
 
    def plot_results(self):
        print('size output', self.output_y.shape, self.output_noise.shape)
        plt.plot(self.output_y[0], label='E')
        plt.plot(self.output_y[1], label='I')
        #plt.plot(self.output_y[2], label='A')
        plt.legend()
        plt.show()
        plt.plot(self.output_y[2], label='A')
        plt.legend()
        plt.show()


    def save_output(self):
        
        f_rate_E = pd.DataFrame(self.output_y[0])
        f_rate_I = pd.DataFrame(self.output_y[1])
        f_rate_A = pd.DataFrame(self.output_y[2])

        mean_frateE = np.mean(f_rate_E, axis=0)
        mean_frateI = np.mean(f_rate_I, axis=0)
        mean_frate = np.mean(np.column_stack((mean_frateI, mean_frateE)), axis=1)

        #print('\nFiring rate Excitatory')
        mouse_dataE = np.array([0, 4.607127, 2.476808, 0, 0, 2.670017, 0, 4.845092, 4.197598, 2.550010, 0, 1.428299, 0, 3.859802])
        mouse_dataI = np.array([0, 10.608670, 6.859976, 0, 0, 6.348029, 0, 10.113730, 7.396667, 7.787373, 0, 4.846161, 0, 7.704969])
        mouse_data_avg = [4.349698, 4.964803, 3.973509, 12.001597, 4.539685, 2.516638, 12.341531, 4.020949, 8.629651, 3.584227, 9.663268, 2.610587, 5.197313, 4.263197]
        mouse_data_awake = np.array([3.167551, 6.128178, 3.164425, 12.686679, 3.984528, 3.040926, 12.146115, 5.614480, 4.976522, 3.015229, 7.420361, 2.926400, 6.624088, 4.435377])
        #print(pd.concat((pd.DataFrame(mean_frateE, columns=['Emodel']), pd.DataFrame(mouse_dataE, columns=['Emouse']), pd.DataFrame(mean_frateI, columns=['Imodel']),pd.DataFrame(mouse_dataI, columns=['Imouse']),
        #                 pd.DataFrame((mean_frateE-mouse_dataE), columns=['Ediff']), pd.DataFrame((mean_frateI - mean_frateE), columns=['I-E'])), axis=1))
        #print('Firing rate all')
        #print(pd.concat((pd.DataFrame(mean_frate, columns=['mean_model']),
        #                 pd.DataFrame(mouse_data_awake, columns=['mean_mouse_awake']),
        #                 pd.DataFrame(mouse_data_awake-mean_frate, columns=['mean_diff_awake']),
        #                 pd.DataFrame(mouse_data_avg, columns=['mean_mouse_sleep']),
        #                 pd.DataFrame(mouse_data_avg-mean_frate, columns=['mean_diff_sleep'])), axis=1))

        f_rate_E.to_csv(op.join(self.output_dir, f'frateE_{self.file_addon}{self.session}.csv'), index=False)
        f_rate_I.to_csv(op.join(self.output_dir, f'frateI_{self.file_addon}{self.session}.csv'), index=False)
        f_rate_A.to_csv(op.join(self.output_dir, f'frateA_{self.file_addon}{self.session}.csv'), index=False)
        


