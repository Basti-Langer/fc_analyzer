def get_hfrs(data: list=[], sort_by: str='Vdc', Zimag_range: list=[0.0015, 0.005], mode: str='intersect',
             min_freq: float=100, max_freq: float=100e3, induct: bool=False, fit_points: float=5,
             cycle: int=None, save: bool=False, save_mode: str='x'):
    '''
    Determine the HFRs of several spectra at once. Based on 'eis_class.get_hfr()'
    
    'data' is a list of eis_class objects
    
    Two possible modes. Both apply a linear regression:
        (1) 'intersect' takes the actual intersect that is characterized by a linear fit
        through the closest points (# given by 'fit_points') to the real (x-) axis.
        (2) 'linfit' takes the intercept of a linear fit through points specified by
        a range of imaginary resistance values ('Zimag_range' in Ohm cm2)
        
    'fit_points' is the number of data points closest to the real axis considered for 'mode=intersect'
        
    'Zimag_range' is the range of considered imaginary impedance (in Ohm cm2) in case of 'mode=linfit'
    
    'induct = True' allows to correct the raw data for the inductivity before calculating
    the HFR (get inductivity from Excel sheet)
    
    'min_freq' (float) is useful to exclude a possible low frequency intercept (LFR)
    
    'max_freq' (float) sets upper frequency limit
        
    'sort_by' is either 'Vdc' to sort the spectra by their (average) DC potential (from high to low)
    or 'Idc' to sort by their (average) DC current (from low to high)
    or 'time' to sort by their aquisition time
    
    'cycle' specifies the cycle number of an AST (for possible lookup)
    
    'save' refers to saving a .csv file with the fitted data
    
    'save_mode' allows to control overwriting of existing fit summaries.
    
    'save_mode=x' means not not overwrite an exisiting file, 'save_mode=w' means to overwrite anyways
    
    INPUT
    data:               list
    sort_by:            str
    mode:               str
    Zimag_range:        list
    fit_points:         float
    min_freq:           float
    max_freq:           float
    induct:             bool
    save:               bool
    save_mode:          str
    
    RETURN
    HFRs:               data frame
    '''   
    import pandas as pd
    import os
    
    # summary data frame
    HFRs = pd.DataFrame(columns=['filename', 't_rel [s]', 'HFR [Ohmcm2]', 'Z_imag range [Ohmcm2]', 'fit_points',
                                 'm (slope)', 'c (y-intercept)', 'method', 'extrapolation', 'induct L [H]',
                                 'induct beta []']) 
    
    # sort data
    if sort_by == 'time':
        data.sort(key=lambda x: os.path.getmtime(x.path)) # sort by creation date (in seconds)
    elif sort_by == 'Vdc':
        data.sort(key=lambda x: x.data['Vdc'].mean(), reverse=True) # sort by potential, high to low
    elif sort_by == 'Idc':
        data.sort(key=lambda x: x.data['Idc'].mean()) # sort by current, low to high
    
    # get time stamp of first data file for t_rel
    t0 = min([os.path.getmtime(d.path) for d in data])
    
    # get HFR of each data file and write into summary file
    for i, v in enumerate(data):
        hfr, linreg, l, extrapol = v.get_hfr(induct=induct, Zimag_range=Zimag_range,
                                   min_freq=min_freq, max_freq=max_freq, mode=mode,
                                   fit_points=fit_points, cycle=cycle, save=True, show=False)

        HFRs.at[i, 'filename'] = v.name
        HFRs.at[i, 't_rel [s]'] = os.path.getmtime(v.path) - t0
        HFRs.at[i, 'HFR [Ohmcm2]'] = hfr
        HFRs.at[i, 'extrapolation'] = extrapol
        HFRs.at[i, 'm (slope)'] = linreg[0]
        HFRs.at[i, 'c (y-intercept)'] = linreg[1]
        HFRs.at[i, 'induct L [H]'] = l[0]
        HFRs.at[i, 'induct beta []'] = l[1]
              
    if mode == 'linfit':
        HFRs['Z_imag range [Ohmcm2]'] = str(Zimag_range)
    elif mode == 'intersect':
        HFRs['fit_points'] = fit_points
        
    HFRs['method'] = mode
        
    if save == True:
        import os
        dirname = os.path.dirname(data[0].path) # path
        basename = os.path.basename(dirname) # only name of direcotory
        mea = data[0].mea
        save_path = os.path.join(dirname, f'B-MEA-{mea}_{basename}_HFRs_{mode}_induct{str(induct)}.csv')
        HFRs.to_csv(save_path, sep='\t', mode=save_mode)
        
    return HFRs

def batch_pc_hfrs(sort_by: str='Vdc', Zimag_range: list=[-0.0025, 0.0025], 
             min_freq: float=100, max_freq: float=100e3, induct: bool=False, 
             mode: str='intersect', fit_points: float=5, cycle: int=None,
             save: bool=False, save_mode: str='x'):
    
    '''
    Finds all polcurves in subdirectories of a selected directory 
    and fits the HFRs corresponding to each polcurve based on 'eis_class.get_hfr()'
    
    Two possible modes. Both apply a linear regression:
        (1) 'intersect' takes the actual intersect that is characterized by a linear fit
        through the closest points (# given by 'fit_points') to the real (x-) axis.
        (2) 'linfit' takes the intercept of a linear fit through points specified by
        a range of imaginary resistance values ('Zimag_range' in Ohm cm2)
        
    'fit_points' is the number of data points closest to the real axis considered for 'mode=intersect'
        
    'Zimag_range' is the range of considered imaginary impedance (in Ohm cm2) in case of 'mode=linfit'
    
    'induct = True' allows to correct the raw data for the inductivity before calculating
    the HFR (get inductivity from Excel sheet)
    
    'min_freq' (float) is useful to exclude a possible low frequency intercept (LFR)
    
    'max_freq' (float) sets upper frequency limit
        
    'sort_by' is either 'Vdc' to sort the spectra by their (average) DC potential (from high to low)
    or 'Idc' to sort by their (average) DC current (from low to high)
    or 'time' to sort by their aquisition time
    
    'save' refers to saving a .csv file with the fitted data (one for each polcurve)
    
    'save_mode' allows to control overwriting of existing fit summaries.
    
    'save_mode=x' means not not overwrite an exisiting file, 'save_mode=w' means to overwrite anyways
    
    INPUT
    data:               list
    sort_by:            str
    mode:               str
    Zimag_range:        list
    fit_points:         float
    min_freq:           float
    max_freq:           float
    induct:             bool
    save:               bool
    save_mode:          str
    
    RETURN
    None
    '''
    import glob as glob
    import os
    from .find_open import read_data
    from .util import select_folder
    
    print('Select directory to find polcurves and caclculate HFRs (keyword="PolCurve").')
    dir0 = select_folder()

    # Find all polcurve directories and select corresponding EIS data
    # Might also find the corresponding data loggers which have similar naming
    # But since in these directories no eis data, no problem
    pc_paths = glob.glob(os.path.join(dir0, '**/*PolCurve*.csv'), recursive=True)
    
    pc_dirs = []
    for i in pc_paths:
        pc_dirs.append(os.path.dirname(i))
    
    # avoid duplicates
    pc_dirs = set(pc_dirs)
    
    # get hfrs for each polcurve
    for i in pc_dirs:
        try:
            print(f'Fitting HFRs for {os.path.basename(i)}')
            eis_paths = glob.glob(os.path.join(i, '*EIS*.DTA'))
            if len(eis_paths) > 0:
                eis_data = read_data(eis_paths, 'eis')
                
                get_hfrs(data=eis_data, sort_by=sort_by, Zimag_range=Zimag_range, 
                    min_freq=min_freq, max_freq=max_freq, induct=induct, mode=mode,
                    fit_points=fit_points, cycle=cycle, save=save, save_mode=save_mode)
        except:
            print("\033[31m", f"Fitting of {os.path.basename(i)} failed", "\033[0m")

def single_freq(data: list=[], freq: int=200, induct: bool=False,
            hfr_corr: bool=False, save: bool=False, show: bool=True,
            t0_sp: int=0, t_shift: float=0, mode: str='phi'):
    '''
    Plot either the phase angle or the modulus at a single frequency of several (corrected) eis spectra
    against the time of data acquisition
    
    'freq' is the frequency of interest. The next closest data point is considered
    
    'induct' sets the inductivity correction
    
    'hfr_corr' applies a hfr correction by locally reading from a HFR file
    
    'save' allows saving the plot and a summary .csv file
    
    'show' refers to the plot
    
    't0_sp' can be used to set the time stamp of a specific spectrum to t=0
    
    't_shift' applies an absolute time shift in seconds
    
    mode is either
    'phi' for phase angle
    'mod' for the modulus
    
    Mind that the inductivity correction should be consistent:
    If one corrects for the inductivity, the HFR should also be based on inductivity corrected impedance
    
    INPUT
    data:       list
    freq:       int
    induct:     bool
    hfr_corr:   bool
    save:       bool
    show:       bool
    t0_sp:      int
    t_shift:    float
    mode:       str
    
    RETURN
    time:       list
    y:          list (either phase angle or modulus)
    Z:          list (complex impedance)               
    '''
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from .lookup import lookup_xls
    from .eis_models import inductor
    from .constants import get_active_area
    
    A = get_active_area() # active area in cm2
    
    # order data by time
    data.sort(key=lambda x: os.path.getmtime(x.path))
    eis_paths = [spectrum.path for spectrum in data]
    
    # correct for time offset
    t0 = os.path.getmtime(eis_paths[t0_sp]) + t_shift
    time = [(os.path.getmtime(p) - t0) for p in eis_paths]  # time of frame in s
    
    # get data
    eis_data = [i.data for i in data]
    names = [i.name for i in data] # list of file names
    
    # narrow down to data with frequency closest to 'freq'
    eis_data = [d.iloc[(d['Freq']-freq).abs().argsort()[0]] for d in eis_data]
    freq_sel = np.real(eis_data[0].loc['Freq']) # actual frequency that is plotted
    Z = [d['Z_complex'] for d in eis_data]
    
    # inductivity correction
    if induct:
        mea = data[0].mea # assuming all spectra from the same MEA
        l = (lookup_xls(mea, 'Inductivity L'),  lookup_xls(mea, 'Inductivity beta')) # (L, beta); L in H
        Z = [z - inductor(freq_sel, l) for z in Z]
        
    # hfr correction
    if hfr_corr:
        from .find_open import read_hfrs
        # read HFRs from local file
        HFRs = read_hfrs(os.path.dirname(data[0].path))
        # select HFRs according to file name
        HFRs_sel = HFRs[HFRs['filename'].isin(names)]
        # Mind units! Substract HFR in Ohm
        HFRs_sel = [(1/A) * float(HFRs_sel['HFR [Ohmcm2]'][HFRs_sel['filename'] == n]) for n in names]
        Z = [z - hfr for z, hfr in zip(Z, HFRs_sel)]
        
    if mode == 'phi':
        fig = plt.figure(f'Phi(f) vs. time')
        # get angle of the corrected impedance
        y = [- np.angle(z, deg=True) for z in Z]
        plt.xlabel(r'$t \, [s]$')
        plt.ylabel(r'$\Phi \, [°]$')
        plt.scatter(time, y, label=f'$f = {round(freq_sel, 1)} Hz$', s=15)
        plt.hlines(y=45, xmin=min(time), xmax=max(time), linestyles='dashed', colors='k')
        
    if mode == 'mod':
        fig = plt.figure(f'|Z|(f) vs. time')
        y = [np.absolute(z) * A for z in Z] # modulus of impedance
        plt.xlabel(r'$t \, [s]$')
        plt.ylabel(r'$\abs{Z} [\Omega]$')
        plt.scatter(time, y, label=f'$f = {round(freq_sel, 1)} Hz$', s=15)
        
    fig.legend()
    
    if save:
        import pandas as pd
        # save plot
        save_path = f'B-MEA-{data[0].mea}_{mode}-vs-time_induct{str(induct)}_hfr-corr{str(hfr_corr)}.png'
        save_path = os.path.join(os.path.dirname(eis_paths[0]), save_path)
        plt.savefig(save_path, format='png')
        
        # save data in .csv
        if mode == 'phi':
            y_name = 'Phi [°]'
        elif mode == 'mod':
            y_name = '|Z| [Ohmcm2]'
        summary_df = pd.DataFrame(columns=['filename', 'f [Hz]', 't [s]', f'{y_name}',
                                           'L [H]', 'beta', 'HFR_corr [Ohmcm2]'])
        summary_df['filename'] = [os.path.basename(path) for path in eis_paths]
        summary_df['f [Hz]'] = freq_sel
        summary_df['t [s]'] = time
        summary_df[f'{y_name}'] = y # y depends on mode keyword
        if induct:
            summary_df['L [H]'] = l[0]
            summary_df['beta'] = l[1]
        if hfr_corr:
            summary_df['HFR_corr [Ohmcm2]'] = HFRs_sel
        
        save_path = f'{os.path.splitext(save_path)[0]}.csv'
        summary_df.to_csv(save_path, sep='\t')
        
    if show:
        plt.show()
        
    return time, y, Z
    