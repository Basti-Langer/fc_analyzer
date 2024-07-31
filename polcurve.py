def get_zeta(current, rp):
    '''
    Get the zeta parameter for a voltage loss analysis
    s. paper DOI: 10.1149/1.2400626
    
    'current' in A/cm2
    
    'rp' in Ohm cm2
    
    INPUT
    current:        float
    rp:             float
    '''
    zeta = (27.524 - 27.528 * 0.9744**((current * rp)/0.07))
    return zeta

class polcurve_class:
    ''''
    Class for polcurve data files
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
        
        with open(path, 'r', encoding='cp1252') as f: 
            lines = f.readlines()
            for i in lines:
                if i.find('Unit') != -1:
                    skiprows = lines.index(i)

        
        pc = pd.read_csv(path, sep=',', encoding='cp1252', skiprows=skiprows+1)
        
       
        pc['E [V]'] = pc['cell_voltage_001']
        pc['i [A/cm2]'] = pc['current'] / pc['cell_active_area']
        self.data = pc.sort_values(by='current', ignore_index=True)
        self.active_area = pc['cell_active_area'][0]
        
        # several parameters that are constant during one polcurve are averaged
        #self.anode_T = pc['temp_anode_endplate'].mean()
        self.cathode_T = pc['temp_cathode_endplate'].mean()
        self.anode_dp = pc['temp_anode_dewpoint_water'].mean()
        self.cathode_dp = pc['temp_cathode_dewpoint_water'].mean()
        self.anode_flow = pc['total_anode_stack_flow'].mean()
        self.cathode_flow = pc['total_cathode_stack_flow'].mean()
        

    def polcurve(self, save: bool=False, show: bool=True):
        '''
        Plot a polarization curve.
        
        INPUT
        save:           bool
        show:           bool
        
        RETURN
        current:        list
        voltage:        list
        '''
        import matplotlib.pyplot as plt
        
        current = self.data['i [A/cm2]']
        voltage = self.data['E [V]']

        fig = plt.figure(f'Polcurve {self.name}') # opens current figure (to plot several into oen figure)
        plt.ylabel(r'$E [V]$')
        plt.xlabel(r'$i \, [A/cm^2]$')
        plt.scatter(current, voltage, label=self.name)
        fig.legend()
        
        if save == True:
            import os
            dirname = os.path.dirname(self.path)
            basename = os.path.basename(dirname)
            save_path = os.path.join((dirname), f'B-MEA-{self.mea}_{basename}_pc.png')
            plt.savefig(save_path)
            
        if show == True:
            plt.show()
            
        plt.close(fig=fig)
            
        return current, voltage
        
    def polcurve_hfr(self, save: bool=False, show: bool=True, save_csv=False):
        '''
        Plot a polarization curve with corresponding HFRs
        HFRs must be available in local summary file (.csv)
        
        INPUT
        save:           bool
        show:           bool
        
        RETURN
        polcurve:       data frame
        '''
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        from .find_open import read_hfrs
        
        polcurve = pd.DataFrame(columns=['i [A/cm2]', 'E [V]', 'HFR [Ohm cm2]'])
        polcurve['i [A/cm2]'] = self.data['i [A/cm2]']
        polcurve['E [V]'] = self.data['E [V]']
        
        HFRs = read_hfrs(os.path.dirname(self.path))
        polcurve['HFR [Ohm cm2]'] = HFRs['HFR [Ohmcm2]']
        
        fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[0.75, 0.25])

        axs[0].set_ylabel(r'$E [V]$')
        axs[1].set_ylabel(r'$R_{HFR} [m \Omega \cdot cm^2]$')
        axs[1].set_xlabel(r'$i \, [A/cm^2]$')
        axs[0].scatter(polcurve['i [A/cm2]'], polcurve['E [V]'], label=self.name)
        axs[1].scatter(polcurve['i [A/cm2]'], polcurve['HFR [Ohm cm2]']*1000) 
        
        if save == True:
            import os
            dirname = os.path.dirname(self.path)
            basename = os.path.basename(dirname)
            save_path = os.path.join((dirname), f'B-MEA-{self.mea}_{basename}_pc-hfr.png')
            plt.savefig(save_path)
        
        if show == True:
            plt.show()
        
        plt.close(fig=fig)
        
        if save_csv:
            import os
            dirname = os.path.dirname(self.path)
            basename = os.path.basename(dirname)
            save_path = os.path.join((dirname), f'B-MEA-{self.mea}_{basename}_pc-summary.csv')
            polcurve.to_csv(save_path, sep='\t')
            
        return polcurve
        
    def tafel(self, mode: str='mass', save: bool=False, show: bool=True, current_range_pt: list=[2, 6], short: bool=True):
        '''
        Standard Tafel analysis of a polcurve
        Takes all the correction factors from the averaging Excel (Error message if not accessible)
        'mode' is either 
            'mass' to calculate the mass activity ('act') in A/g(Pt) or
            'CO' or 
            'HUPD' to calculate the area-normalized activity ('act') in A/cm2(Pt)
            
        'current_range_pt' defines the range (index) of current points of the polcurve that are used for the analysis
        
        'save' and 'show' refer to the plot
        
        'ts' is the Tafel slope in mv/dec
        
        short (SL,30.7.24): if the short current is so low a linear fit is not possible, the short current correction should NOT BE DONE. select false to skip it
        
        INPUT
        mode:               str
        current_range_pt:   list
        save:               bool
        show:               bool
        
        RETURN
        act:                float
        ts:                 float
        '''
        from scipy import stats
        from .lookup import lookup_xls
        from .find_open import read_hfrs
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # read current range points
        a = current_range_pt[0]
        b = current_range_pt[1]
        
        # initialize figure
        fig = plt.figure(f'Tafel {self.name}, mode={mode}')
        plt.ylabel(r'$E_{HFR,\,R_{H+cath}-free}$')
        
        # get correction factors 
        mea = self.mea
        r_e = lookup_xls(mea, 'SHORT') # in Ohmcm2
        if short==True: print(f'R_el =\t {round(r_e)} Ohm cm2')
        if short==False: print('no short correction')
        
        h2x = lookup_xls(mea, 'H2x') # in mA/cm2
        print(f'i_H2x =\t {h2x} mA/cm2')
        h2x = h2x / 1000 # in A/cm2
        
        rp95 = lookup_xls(mea, 'R_H+ (95)') # in mOhm cm2
        print(f'R_p(RH=95%) =\t {rp95} mOhm cm2')
        rp95 = rp95 / 1000 # in Ohm cm2
        
        loading = lookup_xls(mea, 'Loading') # in mg/cm2
        print(f'L_Pt =\t {loading} mg_Pt/cm2')
        loading = loading / 1000 # in g/cm2

        HFRs = read_hfrs(os.path.dirname(self.path))
        hfrs = HFRs['HFR [Ohmcm2]'] # in Ohm cm2
        
        # do corrections
        potential = self.data['E [V]'] # in V

        current = self.data['i [A/cm2]'] # in A/cm2
        zeta = get_zeta(current, rp95) # correction factor

        rp_eff = rp95 / (3 + zeta)

        e_corr = potential + hfrs*current + rp_eff*current
        if short==True:
            i_corr = self.data['i [A/cm2]'] + h2x + self.data['E [V]'] / r_e # in A/cm2
            print('Short current correction is applied')
        elif short==False:
            i_corr = self.data['i [A/cm2]'] + h2x  # in A/cm2
        else:
            print('error with short-boolean')
        
        i_corr = i_corr / loading # in A/g

        # different options for normalization
        if mode == 'mass':
            if short==True: plt.xlabel(r'$i + i_{H2x} + i_{short} \, [A/g]$')
            if short==False: plt.xlabel(r'$i + i_{H2x} \, [A/g]$')
            unit = 'A/g'
            
        elif mode == 'HUPD':
            ecsa = lookup_xls(mea, f'ECSA HUPD') # in m2/g
            print(f'ECSA_HUPD =\t {ecsa} m2/g')
            i_corr = i_corr / ecsa # in A/m2
            if short==True: plt.xlabel(r'$i + i_{H2x} + i_{short} \, [A/g]$')
            if short==False: plt.xlabel(r'$i + i_{H2x} \, [A/g]$')
            unit = 'A/m^2'
        
        elif mode == 'CO':
            ecsa = lookup_xls(mea, f'ECSA CO') # in m2/g
            print(f'ECSA_CO =\t {ecsa} m2/g')
            i_corr = i_corr / ecsa # in A/m2
            if short==True: plt.xlabel(r'$i + i_{H2x} + i_{short} \, [A/g]$')
            if short==False: plt.xlabel(r'$i + i_{H2x} \, [A/g]$')
            unit = 'A/m^2'
        else:
            raise ValueError('select mode: mass, HUPD, or CO')
        
        # linear regression with log x values in given range
        res = stats.linregress(np.log10(i_corr).iloc[a:b], e_corr[a:b]) 
        
        ts = abs(res[0]) # Tafel slope in V/dec
        act = (0.9 - res[1]) / res[0] # activity i.e. current @0.9V in log A/cm2; x = (y-c)/m
        act = 10**act # in A/cm2
        
        # visualization
        plt.scatter(i_corr, e_corr, color='grey')
        plt.scatter(i_corr[a:b], e_corr[a:b], color='r')
        ###########
        i_raw = current/loading
        plt.scatter(i_raw,potential,color='green')
        ###########
        x = [i_corr[a], i_corr[b-1]]
        y1 = res[0] * np.log10(x[0]) + res[1] # y=mx+c
        y2 = res[0] * np.log10(x[1]) + res[1]
        plt.plot(x, [y1, y2], color='r')
        plt.xscale('log')
        plt.title(f'$TS={round(ts*1000, 1)}'+ r'\, mV/dec; \,' + f'MA={round(act, 3)}\,{unit}$')
        plt.legend(['grey+red:corrected polcurve','red:used for fit','green: non-corrected'])
        
        if save:
            import os
            f_name = os.path.splitext(self.name)[0] # kick out '.csv'
            save_path = os.path.join(os.path.dirname(self.path), f'{f_name}_Tafel_mode-{mode}.png')
            plt.savefig(save_path, format='png')
            
        if show:
            plt.show()
        
        plt.close(fig=fig)
        
        return act, ts
            
