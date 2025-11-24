import os
import os.path as op
import math
import numpy as np 
from datetime import datetime
import pandas as pd
import yaml

class SimulationSession():

    def __init__(self, output_dir, nrAreas, filepath_connectivity, settings_file, drn_connect_file, G, S, time_scale, session=''):
        # a session needs parameters and output functions 
        self.nrVars = 3 # E, I and A
        self.nrAreas = nrAreas
        self.initial_cond = np.zeros((self.nrVars, self.nrAreas))
        self.noise_init = np.zeros((2, self.nrAreas))
        self.output_dir = output_dir
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
        extra = f"RateAdj1_sessions"
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
        outname = self.file_addon + self.session + '_stimulation_times.csv'
        stim_times = pd.DataFrame(self.stimulation_times)
        stim_times.to_csv(os.path.join(outdir, outname), index=False)

    
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
        #mean_connect = np.mean(self.c_matrix.to_numpy().flatten())

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
        aux = self.Jee*y[0] - self.Jei*y[1] + self.thetaE_array + n[0] - y[2] + self.G * np.matmul(self.c_matrix, y[0]) - self.I

        for area in np.arange(self.nrAreas):
            if aux[area] <= self.Edesp:
                dy[0][area] = -y[0][area]/self.tauE
            else:
                dy[0][area] = (-y[0][area] + ((aux[area]-self.Edesp)*self.Eslope[area]))/self.tauE
        
        # derivative of I - rate 
        aux = self.Jie*y[0] - self.Jii*y[1] + self.thetaI + n[1] # - self.I
        for area in np.arange(self.nrAreas):
            if aux[area] <= self.Idesp:
                dy[1][area] = -y[1][area]/self.tauI
            else:
                dy[1][area] = (-y[1][area] + self.Islope[area]*(aux[area]-self.Idesp))/self.tauI

        # derivative of A - adaptation rate
        dy[2] = (-y[2]+self.betaE_array*y[0])/self.tauAdapt

        return dy


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

        print('\nFiring rate Excitatory')
        mouse_dataE = np.array([0, 4.607127, 2.476808, 0, 0, 2.670017, 0, 4.845092, 4.197598, 2.550010, 0, 1.428299, 0, 3.859802])
        mouse_dataI = np.array([0, 10.608670, 6.859976, 0, 0, 6.348029, 0, 10.113730, 7.396667, 7.787373, 0, 4.846161, 0, 7.704969])
        mouse_data_avg = [4.349698, 4.964803, 3.973509, 12.001597, 4.539685, 2.516638, 12.341531, 4.020949, 8.629651, 3.584227, 9.663268, 2.610587, 5.197313, 4.263197]
        mouse_data_awake = np.array([3.167551, 6.128178, 3.164425, 12.686679, 3.984528, 3.040926, 12.146115, 5.614480, 4.976522, 3.015229, 7.420361, 2.926400, 6.624088, 4.435377])
        print(pd.concat((pd.DataFrame(mean_frateE, columns=['Emodel']), pd.DataFrame(mouse_dataE, columns=['Emouse']), pd.DataFrame(mean_frateI, columns=['Imodel']),pd.DataFrame(mouse_dataI, columns=['Imouse']),
                         pd.DataFrame((mean_frateE-mouse_dataE), columns=['Ediff']), pd.DataFrame((mean_frateI - mean_frateE), columns=['I-E'])), axis=1))
        print('Firing rate all')
        print(pd.concat((pd.DataFrame(mean_frate, columns=['mean_model']),
                         pd.DataFrame(mouse_data_awake, columns=['mean_mouse_awake']),
                         pd.DataFrame(mouse_data_awake-mean_frate, columns=['mean_diff_awake']),
                         pd.DataFrame(mouse_data_avg, columns=['mean_mouse_sleep']),
                         pd.DataFrame(mouse_data_avg-mean_frate, columns=['mean_diff_sleep'])), axis=1))

        f_rate_E.to_csv(op.join(self.output_dir, f'frateE_{self.file_addon}{self.session}.csv'), index=False)
        f_rate_I.to_csv(op.join(self.output_dir, f'frateI_{self.file_addon}{self.session}.csv'), index=False)
        f_rate_A.to_csv(op.join(self.output_dir, f'frateA_{self.file_addon}{self.session}.csv'), index=False)
        


