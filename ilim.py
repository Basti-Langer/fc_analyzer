class ilim_class:
    '''
    Class for limiting current data
    '''
    def __init__(self, path):
        import os
        import pandas as pd
        import time
        import re
        from .paths import mea_from_path
        
        self.path = path
        self.name = os.path.basename(path)
        date = os.path.getmtime(path)
        self.date = time.ctime(date)
        
        self.mea = mea_from_path(path)
        
        match = re.search(r'(\d+\.\d+)pO2', self.name)
        if match:
            o2_conc_str=match.group(1)
            self.o2_conc=float(o2_conc_str)
        else:
            with open(path, 'r', encoding='cp1252') as f: 
                lines = f.readlines()
                for i in lines:
                    if i.find('percent') != -1:
                        self.o2_conc = i.split(',')
                        self.o2_conc = self.o2_conc[3].split(' ')
                        self.o2_conc = float(self.o2_conc[0])
                        if self.o2_conc ==5:
                            self.o2_conc = 0.5

        with open(path, 'r', encoding='cp1252') as f: 
           lines = f.readlines()
           for i in lines:
                if i.find('Point ID') != -1:
                   skiprows = lines.index(i)
                
                        
        self.data = pd.read_csv(path, sep=',', encoding='cp1252', skiprows=skiprows)
        pc = pd.read_csv(path, sep=',', encoding='cp1252', skiprows=skiprows)
        self.p = pc['pressure_anode_inlet'].mean()
        self.T = pc['temp_cathode_endplate'].mean()
        self.T_dew_point = pc['temp_cathode_dewpoint_water'].mean()
        pc['E [V]'] = pc['cell_voltage_001']
        pc['i [A/cm2]'] = pc['current'] / pc['cell_active_area']
        self.data = pc.sort_values(by='current', ignore_index=True)
        self.active_area = pc['cell_active_area'][0]
    
    
    def get_ilim(self):
        '''
        Extract the limiting current from a given data file
        
        INPUT
        None
        
        RETURN
        ilim:       float
        '''
        from .constants import get_active_area
        
        A = get_active_area()

        # for 0.5% O2 concentration because of H2 evolution ilim is the current measured at 0.15 V
        if (self.o2_conc == 0.5) or (self.o2_conc == 1):
             ilim = self.data['current'][1] / A # in A/cm2
        # for 2% and 4% O2 concentrations because of H2 evolution ilim is the current measured at 0.1 V
        elif (self.o2_conc == 2) or (self.o2_conc == 4) :
             ilim = self.data['current'][2] / A # in A/cm2
             
        # for higher O2 concentration, max current measured at different potentials is ilim
        else:
            ilim = self.data['current'].max() / A # in A/cm2
            

        return ilim
    
def RTO2(data: list=[], save: bool=False, show=True, save_mode: str='x',save_csv: bool=False):
    '''
    Standard limiting current analysis.
    Yields total transport resistance, y-axis intercept and slope.
    No deconvolution into pressure dependent and independent part
    
    'rh' is the relative humidity (0<rh<1)
    
    'save' refers to the plots and the summary data frames
    
    'save_mode=x' means not not overwrite an exisiting file, 'save_mode=w' means to overwrite anyways
    (only for .csv files possible)
    
    'ilim_overview' is a data frame that summarizes the limiting currents from different data files
    
    'Rt_p' is a data frame that summarizes the total transport resistance vs pressure
    
    INPUT
    data:           list
    rh:             float
    save:           bool
    save_mode:      str
    
    RETURN
    ilim_overview:  data frame
    Rt_p:           data frame
    '''
    from scipy import constants, stats
    from .constants import rh_calc, p_H2O_calc
    from .paths import base_paths
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    mea = data[0].mea # all data must be from the same MEA
    dir = os.path.dirname(data[0].path) # directory 
    pattern = base_paths()[3]

    # overview of transport resistance for each O2 concentration
    ilim_overview = pd.DataFrame(columns=['o2_conc [%]', 'ilim [A/cm2]', 'T [°C]', 'RH []', 'p_abs [kPa]', 'c_o2_in [mol/m3]', 'R_t_o2 [s/cm]'])
    
    for i, v in enumerate(data):
        ilim = v.get_ilim() # in A/cm2
        o2_conc = v.o2_conc / 100 # fraction of 1
        p_abs = v.p + 100 # absolute pressure
        T = v.T
        T_dew_point = v.T_dew_point
        F = constants.physical_constants['Faraday constant'][0]
        rh = rh_calc(T,T_dew_point)
        p_h2o = p_H2O_calc(rh,T)
        c_o2_in = (p_abs - p_h2o)*1000 / (constants.R * (273.15+T)) * o2_conc # inlet O2 concentration in mol/m3
        R_t_o2 = 4*F*c_o2_in / ilim *1e-6 # total transport resistance in s/cm
        
        ilim_overview.at[i, 'o2_conc [%]'] = o2_conc
        ilim_overview.at[i, 'ilim [A/cm2]'] = ilim
        ilim_overview.at[i, 'T [°C]'] = T
        ilim_overview.at[i, 'RH []'] = rh
        ilim_overview.at[i, 'p_abs [kPa]'] = p_abs
        ilim_overview.at[i, 'c_o2_in [mol/m3]'] = c_o2_in
        ilim_overview.at[i, 'R_t_o2 [s/cm]'] = R_t_o2

      
    
    # plots
    
    fig1 = plt.figure(f'{pattern}-{mea} O2 transport resistance')
    plt.xlabel(r'$i_{lim} \, [A/cm^2]$')
    plt.ylabel(r'$R_{T} \, [s/cm]$')
    x = ilim_overview['ilim [A/cm2]']
    y = ilim_overview['R_t_o2 [s/cm]']
    plt.scatter(x, y)
    #plt.title(f'c(O2) = {[round(i*100) for i in set_o2]} %; p = {[round(j) for j in set_p]}')
    fig1.legend()

    if save:
        dirname = os.path.dirname(data[0].path)
        basename = os.path.basename(dirname)
        save_path_3 = os.path.join(dir, f'{pattern}{data[0].mea}_{basename}_RTO2.png')
        plt.savefig(save_path_3)
    
    if show:
        plt.show()

    # Create the formatted line to be inserted
    #formatted_value = f'{pattern}{data[0].mea}_{os.path.basename(os.path.dirname(data[0].path))}'
    #formatted_row = pd.DataFrame([[formatted_value] * len(ilim_overview.columns)], columns=ilim_overview.columns)
        
    # Insert the formatted row at the second position
    #if len(ilim_overview) == 0:
        # If the DataFrame is empty, simply add the formatted row
        #ilim_overview = pd.concat([formatted_row], ignore_index=True)
    #else:
        # Split the DataFrame into two parts and concatenate with the formatted row
        #before_position = ilim_overview.iloc[:0]  # Rows before the insertion point
        #after_position = ilim_overview.iloc[0:]   # Rows after the insertion point
        #ilim_overview = pd.concat([before_position, formatted_row, after_position], ignore_index=True)

    # save csv
    if save_csv:
        dirname = os.path.dirname(data[0].path)
        basename = os.path.basename(dirname)
        save_path_1 = os.path.join(dir, f'{pattern}{data[0].mea}_{basename}_RTO2.csv')
        ilim_overview.to_csv(save_path_1, sep='\t', mode=save_mode)
        
        #save_path_2 = os.path.join(dir, f'B-MEA-{mea}_Rt_p.csv')
        #Rt_p.to_csv(save_path_2, sep='\t', mode=save_mode)  

    
    return ilim_overview

def ilim_overview(data: list=[], show: bool=True, save: bool=False, save_csv: bool=False):

    import matplotlib.pyplot as plt
    import pandas as pd
    from .paths import base_paths

    pattern = base_paths()[3]
    fig = plt.figure('ilim_plot')
    plt.ylabel(r'$E [V]$')
    plt.xlabel(r'$i \, [A/cm^2]$')
        
    # initialize data frame
    iterables = [[v.name for v in data], ['i [A/cm2]', 'E [V]']]
    columns = pd.MultiIndex.from_product(iterables)
    length = max([len(v.data['i [A/cm2]']) for v in data]) # check out the longest polcurve to initilaize df
    index = range(length)
    pc_df = pd.DataFrame(columns=columns, index=index)
        
    # Summarize data into one data frame and plot
    for i, v in enumerate(data):
        current = v.data['i [A/cm2]']
        voltage = v.data['E [V]']
        a = current.idxmin() # get index range; current must be sorted from low to high
        b = current.idxmax() 
        pc_df.loc[a:b, (v.name, 'i [A/cm2]')] = current # need to specify index exactly for pandas MultiIndex
        pc_df.loc[a:b, (v.name, 'E [V]')] = voltage
            
        plt.scatter(current, voltage, label=v.o2_conc)
            
    fig.legend()
            
    if save_csv:
        import os
        dirname = os.path.dirname(data[0].path)
        basename = os.path.basename(dirname)
        save_path = os.path.join((dirname), f'{pattern}{data[0].mea}_{basename}_ilim_overview.csv')
        pc_df.to_csv(save_path, sep='\t')
    
    if save == True:
        import os
        dirname = os.path.dirname(data[0].path)
        basename = os.path.basename(dirname)
        save_path = os.path.join((dirname), f'{pattern}{data[0].mea}_{basename}_ilim_overview.png')
        plt.savefig(save_path)
            
    if show:
        plt.show()
        
    plt.close(fig=fig)
        
    return pc_df
        
