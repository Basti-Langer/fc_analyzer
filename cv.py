class cv_class:
    '''
    Class for cyclic voltammetry data
    '''
    def __init__(self, path):
        import os
        import pandas as pd
        import time
        from .paths import mea_from_path
        from .constants import get_active_area
        
        self.path = path
        self.name = os.path.basename(path)
        date = os.path.getmtime(path)
        self.date = time.ctime(date)
        self.mea = mea_from_path(path)
        
        # get skiprows and scan rate
        with open(path, 'r', encoding='cp1252') as f: 
            lines = f.readlines()
            for i in lines:
                if i.find('Pt\tT') != -1:
                    skiprows = lines.index(i) # anchor for the first of the two headers
                if i.find('SCANRATE') != -1:
                    scan_rate = i.split()
                    scan_rate = float(scan_rate[2]) # in V/s
        self.scan_rate = scan_rate
        
        data = pd.read_csv(self.path, sep='\t', 
                                encoding='cp1252', header=[skiprows, skiprows+1])
        data = data.droplevel(level=1, axis=1)
        
        A = get_active_area()
        data['i'] = data['Im'] / A * 1000 # mA/cm2
        
        self.data = data
        
    def cv(self, scan: int=2, rhe_shift: float=0.041):
        '''
        Plot a specific scan of the cv (default is second scan)
        rhe_shift in V. Default assumes 5% H2 in Ar (+0.041 V)
        
        INPUT
        scan:        int
        rhe_shift:    float
        
        RETURN
        voltage:       list
        current        list
        '''
        import matplotlib.pyplot as plt
        
        # RHE shift
        data = self.data
        data['V_corr'] = data['Vf'] + rhe_shift
        
        data_select = data[data['Cycle'] == scan]
        current = data_select['i']
        voltage = data_select['V_corr']
        fig = plt.figure(f'CV {self.name}')
        plt.plot(voltage, current, label=f'scan {scan}')
        plt.xlabel(r'$E \, vs. \, RHE \, [V]$')
        plt.ylabel(r'$i \,[mA/cm^2]$')
        fig.legend()
        plt.show()
        plt.close(fig=fig)
        
        return voltage, current
    
    def raw(self):
        '''
        Plot raw data of CV (without potential correction)
        
        INPUT
        None
        
        RETURN
        voltage:       list
        current        list
        '''
        import matplotlib.pyplot as plt
        
        data = self.data
        current = data['i']
        voltage = data['Vf']
        
        fig = plt.figure(f'Raw CV {self.name}')
        plt.plot(voltage, current)
        plt.xlabel(r'$E \, vs. \, E_{cell} \, [V]$')
        plt.ylabel(r'$i \,[mA/cm^2]$')

        plt.show()
        
    def ecsa_hupd(self, scan: int=2, int_limits: list=[0.095, 0.45],
                  rhe_shift: float=0.041, cycle: int=None, show: bool=True, save: bool=False):
        '''
        Integration of anodic and cathodic HUPD integration for a specific scan
        Integration limits are given as list of lower and upper bound ('int_limits')
        'show' and 'save' refer to the plot
        ECSA is given in m2/g(Pt)
        rhe_shift in V. Default assumes 5% H2 in Ar (+0.041 V)
        
        'cycle': Specify voltage cycle for lookup function
        
        INPUT
        scan:          int
        int_limits:     list
        rhe_shift:      float
        show:           bool
        save:           bool
        
        RETURN
        ecsa_anodic:     float
        ecsa_cathodic:   float
        ecsa_mean:      float
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from .lookup import lookup_xls
        
        # RHE shift
        data = self.data
        data['V_corr'] = data['Vf'] + rhe_shift
        
        # get loading and scan rate
        loading = lookup_xls(self.mea, 'Loading', cycle=cycle) # in mg(Pt)/cm2
        print(f'L_Pt = {loading} mg/cm2')
        scan_rate = self.scan_rate # in V/s
        print(f'v = {scan_rate*1000} mV/s')
        spec_charge = 210 # µC/cm2(Pt)
        
        # select the respective scan
        data_select = data[data['Cycle'] == scan]
        current = data_select['i']
        voltage = data_select['V_corr']
        
        # integration
        int_data = data_select[(data_select['V_corr'] >= int_limits[0]) & (data_select['V_corr'] <= int_limits[1])]
        anodic_i = int_data['i'][int_data['i'] > 0]
        anodic_v = int_data['V_corr'][int_data['i'] > 0]
        cathodic_i = int_data['i'][int_data['i'] < 0]
        cathodic_v = int_data['V_corr'][int_data['i'] < 0]
        
        # get current of upper limit (capacitive current)
        idx_anodic = anodic_v.sub(int_limits[1]).abs().idxmin()
        anodic_capa = anodic_i[idx_anodic]
        
        idx_cathodic = cathodic_v.sub(int_limits[1]).abs().idxmin()
        cathodic_capa = cathodic_i[idx_cathodic]
        
        # anodic integration
        anodic_int = np.trapz(anodic_i, x=anodic_v) - (int_limits[1] - int_limits[0]) * anodic_capa
        ecsa_anodic = anodic_int / (spec_charge * scan_rate * loading) * 1e2 # m2/g(Pt)
        
        # cathodic integration
        cathodic_int = abs(np.trapz(cathodic_i, x=cathodic_v)) - abs((int_limits[1] - int_limits[0]) * cathodic_capa)
        ecsa_cathodic = cathodic_int / (spec_charge * scan_rate * loading) * 1e2 # m2/g(Pt)
        
        ecsa_mean = (ecsa_anodic + ecsa_cathodic) / 2
        
        # visualization
        fig = plt.figure(f'ECSA HUPD {self.name}')
        plt.xlabel(r'$E \, vs. \, RHE [V]$')
        plt.ylabel(r'$i \, [mA/cm^2]$')
        plt.plot(voltage, current, color='k')
        plt.plot(int_data['V_corr'], int_data['i'], color='r')
        plt.fill_between(anodic_v, anodic_i, y2=anodic_capa, color='grey')
        plt.fill_between(cathodic_v, cathodic_i, y2=cathodic_capa, color='grey')
        
        plt.title(f'$ECSA_a = {round(ecsa_anodic,2)}\, m^2/g, \, ECSA_c ={round(ecsa_cathodic, 2)}\, m^2/g, \, ECSA_m ={round(ecsa_mean, 2)}\, m^2/g$')
            
        if save:
            import os
            save_path = f'{os.path.splitext(self.path)[0]}_HUPD.png'
            plt.savefig(save_path, format='png') 
            
        if show:
            plt.show()
        
        plt.close(fig=fig)
        
        return ecsa_anodic, ecsa_cathodic, ecsa_mean
    
    def ecsa_co(self, rhe_shift: float=0.041, cycle=None, pot_range_intersect: list=[0.5, 0.7], int_lim_up: float=None, show: bool=True, save: bool=False):
        '''
        Standard analysis for CO strip measurements
        
        Automatic detection of the integration limit
        Lower integration limit is defined as the intercept of both scans (found between 0.6 V and 0.7 V)
        Upper integration limit is defined as the local minimum between 0.87 V and 0.98 V
        
        'cycle' specifies the voltage cycle to look up the correct loading etc  
        
        rhe_shift in V. Default assumes 5% H2 in Ar (+0.041 V)
        
        'pot_range_intersect' defines the potential range (corrected potential) in which the intersect is searched.
        
        INPUT
        rhe_shift:      float
        cycle:          int
        show:           bool
        save:           bool
        
        RETURN
        ecsa:           float
        int_limits      list
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from .lookup import lookup_xls
        from .paths import rh_from_filename
        
        # RHE shift
        data = self.data
        data['V_corr'] = data['Vf'] + rhe_shift
         
        # get loading, scan rate and RH
        loading = lookup_xls(self.mea, 'Loading', cycle=cycle) # in mg(Pt)/cm2
        scan_rate = self.scan_rate # in V/s
        spec_charge = 210 # µC/cm2(Pt)
        rh = rh_from_filename(self.name)
        
        # cycles
        strip_cv = data[data['Cycle'] == 0] # 1st scan
        base_cv = data[data['Cycle'] == 1] # 2nd scanimport
        
        # only anodic current 
        strip_cv_a = strip_cv[strip_cv['i'] > 0]
        base_cv_a = base_cv[base_cv['i'] > 0]
        
        # Find minimum and maximum integration limits
        strip_cv_b = strip_cv_a[(strip_cv_a['V_corr'] >= pot_range_intersect[0]) & (strip_cv_a['V_corr'] <= pot_range_intersect[1])]
        base_cv_interpolate = np.interp(strip_cv_b['V_corr'], 
                                        base_cv_a['V_corr'], base_cv_a['i'], period=None)
        distances = np.abs(strip_cv_b['i']-base_cv_interpolate)
        min_index = np.argmin(distances)
        int_lim_low = strip_cv_b['V_corr'].values[min_index]
        
        # filter for the upper integration limits
        if isinstance(int_lim_up, float): # argument is only accepted if upper integration limit is given as float
            pass
        else:
            mask_up = ((strip_cv['V_corr'] >= 0.87) & (strip_cv['V_corr'] <= 0.98))
            y_filtered_up = strip_cv_a.loc[mask_up, 'i']
            x_filtered_up = strip_cv_a.loc[mask_up, 'V_corr']
            min_index = np.argmin(y_filtered_up)
            int_lim_up = x_filtered_up.iloc[min_index]
        
        int_limits = [int_lim_low, int_lim_up]
        
        # Find indices for integration limits
        idx_min_strip = (strip_cv_a['V_corr'] - int_limits[0]).abs().idxmin() # mind different indices (-> take values); must have same length
        idx_max_strip = (strip_cv_a['V_corr'] - int_limits[1]).abs().idxmin()
        idx_min_base = (base_cv_a['V_corr'] - int_limits[0]).abs().idxmin()
        idx_max_base = (base_cv_a['V_corr'] - int_limits[1]).abs().idxmin()
        
        # integration
        int = np.trapz(strip_cv_a.loc[idx_min_strip:idx_max_strip, 'i'], 
                       x=strip_cv_a.loc[idx_min_strip:idx_max_strip, 'V_corr'])
        base_int = np.trapz(base_cv_a.loc[idx_min_base:idx_max_base, 'i'], 
                            x=base_cv_a.loc[idx_min_base:idx_max_base, 'V_corr'])
        strip_int = int - base_int
        
        ecsa = 0.5 * strip_int / (spec_charge * scan_rate * loading) * 1e2 # m2/g(Pt)
        
        # visualization
        fig = plt.figure(f'ECSA CO-strip {self.name}')
        plt.xlabel(r'$E \, vs. \, RHE \, [V]$')
        plt.ylabel(r'$i \, [mA/cm^2]$')
        plt.plot(strip_cv['V_corr'], strip_cv['i'], color='grey', label='strip')
        plt.plot(strip_cv.loc[idx_min_strip:idx_max_strip,'V_corr'], 
                 strip_cv.loc[idx_min_strip:idx_max_strip, 'i'], color='red')
        plt.plot(base_cv['V_corr'], base_cv['i'], color='k', label='base')
        plt.fill_between(strip_cv_a.loc[idx_min_strip:idx_max_strip, 'V_corr'], 
                         strip_cv_a.loc[idx_min_strip:idx_max_strip, 'i'], 
                         y2=base_cv_a.loc[idx_min_base:idx_max_base, 'i'], 
                         color='grey')
        
        plt.title(f'$ECSA = {round(ecsa,2)}\, m^2/g, \, {round(int_limits[0], 3)} \, V < E < {round(int_limits[1], 3)}\, V$')
        plt.legend()
        
        if save:
            import os
            save_path = os.path.join(f'{os.path.splitext(self.path)[0]}_CO-strip.png')
            plt.savefig(save_path, format='png')

        if show:
            plt.show()
        
        plt.close(fig=fig)
        
        return ecsa, int_limits, rh
    
    def ecsa_co_hold(self, upper_t: float=40, rhe_shift: float=0.041, pot_range_intersect: list=[0.5, 0.8],
                     cycle: int=None, show: bool=True, save: bool=False):
        '''
        Standard analysis for CO strip RH dependend utilization measurements
        
        Automatic detection of the integration limit
        Lower integration limit is defined as the intercept of both cycles (found between 0.6 V and 0.7 V)
        Upper integration limit can be varied, standard: 40 s
        
        rhe_shift in V. Default assumes 5% H2 in Ar (+0.041 V)
        
        'pot_range_intersect' defines the potential range (corrected potential) in which the intersect is searched.
        
        INPUT
        rhe_shift:      float
        show:           bool
        save:           bool
        
        RETURN
        ecsa:           float
        int_limits      list
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from .lookup import lookup_xls
        from .paths import rh_from_filename
        
        # RHE shift
        data = self.data
        data['V_corr'] = data['Vf'] + rhe_shift
        
        # get loading (independent of voltage cycles) and RH
        loading = lookup_xls(self.mea, 'Loading', cycle=cycle) # in mg(Pt)/cm2
        spec_charge = 210 # µC/cm2(Pt)
        rh = rh_from_filename(self.name)
        
        # cv scans
        strip_cv = data[data['Cycle'] == 0] # 1st scan
        base_cv = data[data['Cycle'] == 1] # 2nd scan
        
        # only anodic current 
        strip_cv_a = strip_cv[strip_cv['i'] > 0]
        base_cv_a = base_cv[base_cv['i'] > 0]

        # time correction of base_cv
        base_cv_a['T_corr'] = base_cv_a['T'] - max(strip_cv['T'])
        
        # Find minimum and maximum integration limits on timescale
        strip_cv_intersect = strip_cv_a[(strip_cv_a['V_corr'] >= pot_range_intersect[0]) & (strip_cv_a['V_corr'] <= pot_range_intersect[1])]
        base_cv_interpolate = np.interp(strip_cv_intersect['T'], base_cv_a['T_corr'],
                                        base_cv_a['i'])
        distances = np.abs(strip_cv_intersect['i'] - base_cv_interpolate)
        index_intersect = np.argmin(distances) # intersection by minimal distance
        lower_t = strip_cv_intersect['T'].values[index_intersect]
        
        int_limits = [lower_t, upper_t]
        
        # Find indices for integration limits
        idx_min_strip = (strip_cv_a['T'] - int_limits[0]).abs().idxmin() # mind different indices (-> take values); must have same length
        idx_max_strip = (strip_cv_a['T'] - int_limits[1]).abs().idxmin()
        idx_min_base = (base_cv_a['T_corr'] - int_limits[0]).abs().idxmin()
        idx_max_base = (base_cv_a['T_corr'] - int_limits[1]).abs().idxmin()
        
        # integration with base_cv (interpolation) as baseline
        int = np.trapz(strip_cv_a.loc[idx_min_strip:idx_max_strip, 'i'], 
                       x=strip_cv_a.loc[idx_min_strip:idx_max_strip, 'T'])
        base_int = np.trapz(base_cv_a.loc[idx_min_base:idx_max_base, 'i'], 
                            x=base_cv_a.loc[idx_min_base:idx_max_base, 'T_corr'])
        strip_int = int - base_int
        
        ecsa = 0.5 * strip_int / (spec_charge * loading) * 1e2 # m2/g(Pt)
        
        # visualization
        fig = plt.figure(f'ECSA CO-strip {self.name}' f'RH = {rh}')
        plt.xlabel(r'$t \, [s]$')
        plt.ylabel(r'$i \, [mA/cm^2]$')
        plt.xlim(-2, upper_t + 2)
        plt.plot(strip_cv_a['T'], strip_cv_a['i'], color='grey', label='strip')
        plt.plot(strip_cv_a.loc[idx_min_strip:idx_max_strip,'T'], 
                 strip_cv_a.loc[idx_min_strip:idx_max_strip, 'i'], color='red')
        plt.plot(base_cv_a['T_corr'], base_cv_a['i'], color='k', label='base')
        plt.fill_between(strip_cv_a.loc[idx_min_strip:idx_max_strip, 'T'], 
                         strip_cv_a.loc[idx_min_strip:idx_max_strip, 'i'], 
                         y2=base_cv_a.loc[idx_min_base:idx_max_base, 'i'], 
                         color='grey')
        
        plt.title(f'$ECSA = {round(ecsa,2)}\, m^2/g, \, {round(lower_t, 3)} \, s < t < {round(upper_t, 3)}\, s\, , RH = {rh*100} \%$')
        plt.legend()
        
        if save:
            import os
            save_path = f'{os.path.splitext(self.path)[0]}_CO-strip_Utilization.png'
            plt.savefig(save_path, format='png')

        if show:
            plt.show()
        
        plt.close(fig=fig)

        return ecsa, int_limits, rh
    
def ecsa_hupd_avg(data: list=[], scan: int=2, int_limits: list=[0.095, 0.45],
                  rhe_shift: float=0.041, cycle: int=None, show: bool=True, save: bool=False):   
    '''
    Analyze multiple CVs by HUPD integration at once and average
    Average of all anodic and cathodic charges
    
    'cycle' can be used to specify the voltage cycle for the lookup function

    INPUT
    scan:          int
    int_limits:     list
    rhe_shift:      float
    show:           bool
    save:           bool
    
    RETURN
    a_mean:         float
    c_mean:         float
    m_mean:         float
    '''
    import pandas as pd
    
    # write results into lists and average
    cv_summary = pd.DataFrame(columns=['filename', 'int_limits [V vs. RHE]', 'ECSA_a [m2/g]',
                                       'ECSA_c [m2/g]', 'ECSA_mean [m2/g]', 'total average [m2/g]'])
    for i, v in enumerate(data):
        a, c, m = v.ecsa_hupd(scan=scan, int_limits=int_limits, cycle=cycle,
                              rhe_shift=rhe_shift, show=show, save=save)
        cv_summary.at[i, 'filename'] = v.name
        cv_summary.at[i, 'int_limits [V vs. RHE]'] = int_limits
        cv_summary.at[i, 'ECSA_a [m2/g]'] = a
        cv_summary.at[i, 'ECSA_c [m2/g]'] = c
        cv_summary.at[i, 'ECSA_mean [m2/g]'] = m

    avg = cv_summary['ECSA_mean [m2/g]'].mean()
    cv_summary['total average [m2/g]'] = avg
    
    if save:
        import os
        save_path = os.path.join(os.path.dirname(data[0].path), 'cv_summary.csv')
        cv_summary.to_csv(save_path, sep='\t', index=False)
        
    return avg

def ecsa_co_util(data: list=[], upper_t: float=40, rhe_shift: float=0.041, pot_range_intersect: list=[0.5, 0.8],
                mode: str='hold', cycle=None, show: bool=True, save: bool=False):   
    '''
    Analyze multiple CVs at various RH.
    CO-Strip experiment can be selected with mode='hold' (calls ecsa_co) or mode='continous' (calls ecsa_co_hold)

    INPUT
    upper_t:         float
    rhe_shift:       float
    mode:            str
    show:            bool
    save:            bool

    RETURN
    cv_summary:      list
    '''
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    # write results into lists and average
    cv_summary = pd.DataFrame(columns=['filename', 'RH [%]', 'ECSA [m2/g]', 'Integration limits', 'Utilization'])

    for i, v in enumerate(data):
        if mode == 'hold':
            ecsa, int_limits, rh = v.ecsa_co_hold(upper_t=upper_t, rhe_shift=rhe_shift,
                                                  cycle=cycle, pot_range_intersect=pot_range_intersect, show=show, save=save)
        elif mode == 'continous':
            ecsa, int_limits, rh = v.ecsa_co(rhe_shift=rhe_shift, cycle=cycle, pot_range_intersect=pot_range_intersect, show=show, save=save)
        
        cv_summary.at[i, 'filename'] = v.name
        cv_summary.at[i, 'RH [%]'] = rh
        cv_summary.at[i, 'ECSA [m2/g]'] = ecsa
        cv_summary.at[i, 'Integration limits'] = str(int_limits)
    
    for i in cv_summary.index:
        max_ECSA = cv_summary['ECSA [m2/g]'].max()
        cv_summary.at[i, 'Utilization'] = cv_summary.at[i, 'ECSA [m2/g]'] / max_ECSA

    # plot
    fig = plt.figure(f'Utilization MEA {data[0].mea}')
    plt.xlabel(r'$RH [\%]$')
    plt.ylabel(r'$ECSA_{RH} / ECSA_{RH=100\%}$')
    plt.plot(cv_summary['RH [%]'], cv_summary['Utilization'], color='grey')
    plt.scatter(cv_summary['RH [%]'], cv_summary['Utilization'], color='k')
    
    if save:
        import os
        save_path = os.path.join(os.path.dirname(data[0].path), 'rh_dep_co_strip.png')
        plt.savefig(save_path, format='png')
        
        save_path = os.path.join(os.path.dirname(data[0].path), 'rh_dep_co_strip.csv')
        cv_summary.to_csv(save_path, sep='\t', index=False)    
    if show:
        
        plt.show()

    plt.close(fig=fig)
    
    return cv_summary
        
def cv_comp(data: list=[], rhe_shift: float=0.041, scan: int=2, show: bool=True):
    '''
    Compare several CVs
    
    rhe_shift in V. Default assumes 5% H2 in Ar (+0.041 V)
    
    INPUT:
    data:       list
    rhe_shift:  float
    scan:       int
    show:       bool
    
    RETURN
    -
    '''
    import matplotlib.pyplot as plt

    fig = plt.figure(f'Compare CVs (scan = {scan})')
    plt.xlabel(r'$E \, vs. \, RHE \, [V]$')
    plt.ylabel(r'$i \,[mA/cm^2]$')
    
    for i in data:
        data = i.data
        data['V_corr'] = data['Vf'] + rhe_shift
        
        data_select = data[data['Cycle'] == scan]
        current = data_select['i']
        voltage = data_select['V_corr']
        plt.plot(voltage, current, label=i.name)
        
    fig.legend()
        
    if show:
        plt.show()
    plt.close(fig=fig)