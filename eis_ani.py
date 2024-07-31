def ani_eis(data: list=[], freq_range: list=[0, 1e6], induct: bool=False,
            hfr_corr: bool=False, mode: str='nyquist',
            save: bool=False, show: bool=False,
            t0_sp: int=0, t_shift: float=0, fps: float=3):
    '''
    Creates a video (.gif) of eis data in Nyquist representation in the sequence of
    their acquisition time. Generally, the saving time of the data file is considered
    as time stamp.
    
    'freq_range=[a, b]' defines the lower (a) and upper (b) limit of considered frequencies
    
    'induct=True' allows to correct all spectra for their inductivity prior to plotting them
    
    'hfr_corr=True' allows to correct all spectra for their HFR
    HFR values must be available from local file
    Filenames are matched to select the correct HFR
    
    'mode' is either 'nyquist' or 'bode' and defines x- and y-axis
    
    't0_sp' is the number (index) of the spetrum that is set as zero time.
    A t0_sp > 0 results into negative times of the first spectra
    
    't_shift' is a absolute shift of the relative times in seconds
    
    'fps' defines the frames per second of the video
    
    'show' and 'save' refer to the video
    
    INPUT
    data:           list
    freq_range:     list
    induct:         bool
    hfr_corr:       bool
    mode:           str
    t0_sp:          int
    t_shift:        float
    fps:            float
    save:           bool
    show:           bool
    
    RETURN
    None
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    import os
    from .lookup import lookup_xls
    from .eis_models import inductor
    from .constants import get_active_area
    
    # order data by time
    data.sort(key=lambda x: os.path.getmtime(x.path))
    eis_paths = [spectrum.path for spectrum in data]
    
    t0 = os.path.getmtime(eis_paths[t0_sp]) + t_shift # e.g. 3 spectra before interrupt -> take time from 2nd spectrum + 1s
    
    # initialize plot
    fig, ax = plt.subplots()

    # plot blank legend
    l0 = ax.scatter([], [], label='$\Delta t$', s=15, marker='s', c='w') # emtpy plot for legend generation
    leg = ax.legend(loc='upper left')
    
    # apply freq_range, induct, and hfr_corr condition
    eis_data = [i.data for i in data] # list of data frames
    names = [i.name for i in data] # list of file names
    data_sel = [d[(d['Freq'] > freq_range[0]) & (d['Freq'] < freq_range[1])] for d in eis_data]
    Z = [d['Z_complex'] for d in data_sel]
    freqs = [d['Freq'] for d in data_sel]
    
    A = get_active_area() # active area in cm2
    
    if induct:
        mea = data[0].mea # assuming all spectra from the same MEA
        l = (lookup_xls(mea, 'Inductivity L'),  lookup_xls(mea, 'Inductivity beta')) # (L, beta); L in H
        Z = [z - inductor(f, l) for z, f in zip(Z, freqs)]

    if hfr_corr:
        from .find_open import read_hfrs
        # read HFRs from local file
        HFRs = read_hfrs(os.path.dirname(data[0].path))
        # select HFRs according to file name
        HFRs_sel = HFRs[HFRs['filename'].isin(names)]
        # Mind units! Substract HFR in Ohm
        Z = [z - (1/A) * float(HFRs_sel['HFR [Ohmcm2]'][HFRs_sel['filename'] == n]) for z, n in zip(Z, names)]

    # set unit Ohm cm2
    Z = [z * A for z in Z]
    
    # colors for plot
    cmap = plt.cm.get_cmap('hsv')
    cspace = np.linspace(0, 1, num=len(Z))
    
    if mode == 'nyquist':
        # define plot
        ax.set_xlabel(r'$Z_{real} \, [\Omega cm^2]$')
        ax.set_ylabel(r'$-Z_{imag} \, [\Omega cm^2]$')
        ax.set_aspect('equal')
    
        # get extrema of all data for plot scaling
        x = [np.real(z) for z in Z]
        x_min = min([i.min() for i in x])
        x_max = max([i.max() for i in x])
        
        y = [np.imag(z) for z in Z]
        y_min = min([i.min() for i in y]) 
        y_max = max([i.max() for i in y]) 
        
        # set axis limits with 0.05 Ohm cm2 space from extreme points
        #ax.set_xlim(0.05, 0.08)
        #ax.set_ylim(-0.01, 0.01)
        ax.set_xlim(x_min - 0.05, x_max + 0.05)
        ax.set_ylim(-y_max - 0.05, -y_min + 0.05)

        def animate_1(i):
            time = os.path.getmtime(eis_paths[i]) - t0 # time of frame in s
            time = round(time, 0)
            abs_time = abs(time) # divmode not useful for negative times
            hours, remainder = divmod(abs_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            if time < 0: # artifically add the sign to display negative times
                sign = '-'
            else:
                sign = ''
                
            # possible to use plt.cla() to avoid plotting everthing on top
            nyquist = ax.scatter(x[i], -y[i], marker='s', s=12, color=cmap(cspace[i]))
            
            # reset the legend text with time stamp in each frame
            if hours == 0:
                leg.get_texts()[0].set_text(f'$\Delta t$ = {sign}{minutes} min {seconds} s')
            else:
                leg.get_texts()[0].set_text(f'$\Delta t$ = {sign}{hours} h {minutes} min {seconds} s')
            
            return nyquist

    if mode == 'bode':
        # define plot
        ax.set_xlabel(r'$f \, [Hz]$')
        ax.set_ylabel(r'$\Phi [°]$')
        ax.set_xscale('log')
        
        # get extrema of all data for plot scaling
        x = freqs
        x_min = min([i.min() for i in x])
        x_max = max([i.max() for i in x])
        
        # minus angle since we think 180 shifted in a Nayquist plot
        y = [- np.angle(z, deg=True) for z in Z]
        y_min = min([i.min() for i in y]) 
        y_max = max([i.max() for i in y]) 
        
        # set axis limits
        ax.set_xlim(x_min*0.5, x_max*2)
        ax.set_ylim(y_min-5, y_max+5) 
        ax.hlines(y=45, xmin=x_min, xmax=x_max, linestyle='dashed', color='k')

        def animate_1(i):
            time = os.path.getmtime(eis_paths[i]) - t0 # time of frame in s
            time = round(time, 0)
            abs_time = abs(time) # divmode not useful for negative times
            hours, remainder = divmod(abs_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            if time < 0: # artifically add the sign to display negative times
                sign = '-'
            else:
                sign = ''
                
            # possible to use plt.cla() to avoid plotting everthing on top
            bode = ax.scatter(x[i], y[i], marker='s', s=15, color=cmap(cspace[i]))
            
            # reset the legend text with time stamp in each frame
            if hours == 0:
                leg.get_texts()[0].set_text(f'$\Delta t$ = {sign}{minutes} min {seconds} s')
            else:
                leg.get_texts()[0].set_text(f'$\Delta t$ = {sign}{hours} h {minutes} min {seconds} s')
            
            return bode

    # interval is time between frames in ms
    ani_eis = animation.FuncAnimation(fig, animate_1, interval=1000/fps, repeat=False, frames=len(Z))
        
    
    if save:
        import os
        dirname = os.path.dirname(data[0].path)
        name = f'B-MEA-{data[0].mea}_ani-eis_{mode}_induct{str(induct)}_hfr-corr{str(hfr_corr)}.gif'
        save_path = os.path.join(dirname, name)
        writegif = animation.PillowWriter(fps=fps)
        ani_eis.save(save_path, writer=writegif)

    if show:
        plt.show()
