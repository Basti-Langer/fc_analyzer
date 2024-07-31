class chrono_class:
    '''
    Class for chronoamperometry or chronopotentiometry data
    Based on Gamry .DTA files
    '''
    def __init__(self, path):
        import os
        import pandas as pd
        import time
        from .paths import mea_from_path
        
        self.path = path
        self.name = os.path.basename(path)
        date = os.path.getmtime(path)
        self.date = time.ctime(date)
        self.mea = mea_from_path(path)
        
        # Find skiprows
        with open(path, 'r', encoding='cp1252') as f: 
            lines = f.readlines()
            for i in lines:
                if i.find('Curve\tTABLE') != -1:
                    skiprows = lines.index(i) # anchor for the first of the two headers

        data = pd.read_csv(self.path, sep='\t', 
                                encoding='cp1252', header=[skiprows+1, skiprows+2])
        data = data.droplevel(level=1, axis=1)
        self.data = data
        
    def raw(self):
        '''
        Display the raw data of the chrono experiment
        
        INPUT
        None
        
        RETURN
        time:           list
        current:        list
        voltage:        list
        '''
        import matplotlib.pyplot as plt
        
        time = self.data['T']
        current = self.data['Im']
        voltage =self.voltage['Vf']
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylabel(r'$E\, [V]$')
        ax2.set_ylabel(r'$i\, [A/cm2]$')
        ax1.set_xlabel(r'$t\,[s]$')
        
        ax2.plot(time, current, color='r', label='current')
        ax1.plot(time, voltage, color='b', label='voltage')
        
        fig.legend()
        plt.show()
        plt.close(fig=fig)
        
        return time, current, voltage
        
    def h2x(self, voltage: list=[0.3, 0.4, 0.5, 0.6, 0.7], show: bool=True, save: bool=False, short: bool=True):
        '''
        Standard analysis for the hydrogen crossover
        The potentials used for the analysis are input parameters
        Current average is performed over the last 30s (of typically 120s hold time)
        'show' and 'save' refer to the plot
        'h2x' in mA/cm2 and 'r_electric' in kOhm cm2
        
        introduced 30.07.24(SL): boolean for short circuit: if True: linear fit to IV curve to get y-intercept and slope
                                                            if False: short current too small to fit; take no slope; h2x is average over current datapoints
        
        INPUT
        voltage:        list of floats
        show:           bool
        save:           bool
        
        RETURN
        h2x:            float
        r_electric:      float
        '''
        from scipy import stats
        import pandas as pd
        import matplotlib.pyplot as plt
        from .constants import get_active_area
        import numpy as np
        
        A = get_active_area()
        
        # check whether potentials were actually measured
        sig = self.data['Sig']
        set_volt = sig.unique()
        set_volt = list(set_volt)
        
        # exclude with a hold time of less than 5s (assume 1 data point per second)
        set_volt = [i for i in set_volt if sig.value_counts().loc[i] > 5]
        
        # check whether all voltages given in the input can be found as set voltage
        # create summary data frame    
        if all(v in set_volt for v in voltage):
            avg_voltage = pd.DataFrame(columns=['avg_v', 'avg_i', 'select'])
                
            for i,v in enumerate(set_volt):
                sub = self.data[self.data['Sig'] == v]
                avg_i = sub.iloc[-30:]['Im'].mean()
                avg_v = sub.iloc[-30:]['Vf'].mean()
                avg_voltage.at[i, 'avg_i'] = avg_i/A # in A/cm2
                avg_voltage.at[i, 'avg_v'] = avg_v
                if v in voltage:
                    select = True
                else:
                    select = False
                avg_voltage.at[i, 'select'] = select

            # linear regression
            cond = avg_voltage['select'] == True
            fit_data = avg_voltage[cond]
            
            res = stats.linregress(list(fit_data['avg_v']), list(fit_data['avg_i']))
            
            if short==True: h2x = res[1] # in A/cm2
            if short==False: h2x = fit_data['avg_i'].mean()
            if short==True: r_electric = 1 / res[0] # in Ohm cm2
            if short==False: r_electric = np.inf
            # plot
            fig = plt.figure(f'H2x {self.name}')
            plt.xlabel(r'$E \, [V]$')
            plt.ylabel(r'$i \, [mA/cm^2]$')
            plt.scatter(avg_voltage['avg_v'], avg_voltage['avg_i']*1000, color='grey')
            plt.scatter(avg_voltage['avg_v'][cond], avg_voltage['avg_i'][cond]*1000)
            x = [voltage[0], voltage[-1]]
            if short==True:
                y1 = (res[0] * x[0] + res[1])*1000 # y=mx+c ; in mA/cm2
                y2 = (res[0] * x[1] + res[1])*1000
                plt.plot(x, [y1, y2], color='r')
                print('linear fit')
                
            if short==False:
                y1=h2x*1000
                y2=h2x*1000
                plt.plot(x, [y1, y2], color='r')
                print('h2x from average')
            
            plt.title(r'$i_{H2x}=$' + str(round(h2x*1000, 2)) + r'$\, \mathrm{mA/cm^2},\, R_e=$' + str(round(r_electric/1000, 2)) + r'$\, \mathrm{k\Omega \cdot cm^2}$')
            
            if save:
                import os
                save_path = os.path.join(os.path.dirname(self.path), f'{self.name}_H2x.png')
                plt.savefig(save_path)
            
            if show:
                plt.show()
            
        else:
            raise ValueError(f'Adapt voltage parameter \n Measured: {set_volt}')

        plt.close(fig=fig)
          
        return h2x*1000, r_electric*1000 # in mA/cm2 and Ohm cm2
    
def h2x_avg(data: list=[], voltage: list=[0.3, 0.4, 0.5, 0.6, 0.7], show: bool=True, save: bool=False, short: bool=True):
    '''
    Perform multiple h2x analyses at the same time and average
    'data' must belong to chrono class
    'show' and 'save' refer to shwo and save the plot of each file for itself
    'h2x_mean' in mA/cm2 and 'r_electric_mean' in Ohm cm2
    
    INPUT
        data:           list of chrono objects
        voltage:        list of floats
        show:           bool
        save:           bool
        
    RETURN
        h2x_mean:           float
        r_electric_mean:    float
    '''
    import pandas as pd
    
    h2x_summary = pd.DataFrame(columns=['filename', 'voltage', 'H2x [mA/cm2]',
                                        'R_el [kOhm cm2]', 'H2x_mean [mA/cm2]',
                                        'R_el_mean [kOhm cm2]'])
    for i, v in enumerate(data):
        h, r = v.h2x(voltage=voltage, show=show, save=save, short=short)
        h2x_summary.at[i, 'filename'] = v.name
        h2x_summary.at[i, 'voltage'] = voltage
        h2x_summary.at[i, 'H2x [mA/cm2]'] = h
        h2x_summary.at[i, 'R_el [kOhm cm2]'] = r/1000
    
    avg_h = h2x_summary['H2x [mA/cm2]'].mean()
    h2x_summary['H2x_mean [mA/cm2]'] = avg_h
    avg_r = h2x_summary['R_el [kOhm cm2]'].mean()
    h2x_summary['R_el_mean [kOhm cm2]'] = avg_r
    
    if save:
        import os
        save_path = os.path.join(os.path.dirname(data[0].path), 'h2x_summary.csv')
        h2x_summary.to_csv(save_path, sep='\t', index=False)
    
    return avg_h, avg_r # in mA/cm2 and Ohm cm2
    
