def polcurve_hfr_comp(data: list=[], show: bool=True, save_csv: bool=False):
    '''
    Compare several polcurves with HFR
    The HFR is read from a local summary file (.csv)
    
    'data' is a list of polcurve objects
    
    'show' refers to the plot
    
    'save_csv' allows to save a compact summary of the data of several polcurves
    The file explorer allows to adapt the saving path
    
    INPUT
    data:           list
    show:           bool
    save_csv:       bool
    
    RETURN
    pc_df:          data frame
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    from .find_open import read_hfrs
    import os
    
    fig, axs = plt.subplots(2, 1, sharex = True, height_ratios=[0.75, 0.25])
    axs[0].set_ylabel(r'$E [V]$')
    axs[1].set_ylabel(r'$R_{HFR} [m \Omega \cdot cm^2]$')
    axs[1].set_xlabel(r'$i \, [A/cm^2]$')
    
    # initialize data frame
    iterables = [[v.name for v in data], ['i [A/cm2]', 'E [V]', 'HFR [Ohmcm2]']]
    columns = pd.MultiIndex.from_product(iterables)
    length = max([len(v.data['i [A/cm2]']) for v in data]) # check out the longest polcurve to initilaize df
    index = range(length)
    pc_df = pd.DataFrame(columns=columns, index=index)
    
    # Summarize data into one data frame and plot
    for i, v in enumerate(data):
        current = v.data['i [A/cm2]']
        voltage = v.data['E [V]']
        HFRs = read_hfrs(os.path.dirname(v.path))
        hfrs = HFRs['HFR [Ohmcm2]']
        
        a = current.idxmin() # get index range; current must be sorted from low to high
        b = current.idxmax() 
        
        pc_df.loc[a:b, (v.name, 'i [A/cm2]')] = current
        pc_df.loc[a:b, (v.name, 'E [V]')] = voltage
        pc_df.loc[a:b, (v.name, 'HFR [Ohmcm2]')] = hfrs
        
        hfrs = hfrs*1000
        
        axs[0].scatter(current, voltage, label=v.name)
        axs[1].scatter(current, hfrs) 
        
    fig.legend()
        
    if save_csv:
        from tkinter import filedialog
        from .paths import base_paths
        import os
        data_dir = base_paths()[1]
        data_dir = os.path.join(data_dir, '_polcurve_csv_export/')
        save_path = filedialog.asksaveasfilename(defaultextension='.csv', initialdir=data_dir)
        pc_df.to_csv(save_path, sep='\t')
        
    if show:
        plt.show()
    
    plt.close(fig=fig)
    
    return pc_df 
    
def polcurve_comp(data: list=[], show: bool=True, save_csv: bool=False):
    '''
    Compare several polcurves (without HFR)
    
    'data' is a list of polcurve objects
    
    'show' refers to the plot
    
    'save_csv' allows to save a compact summary of the data of several polcurves
    The file explorer allows to adapt the saving path
    
    INPUT
    data:           list
    show:           bool
    save_csv:       bool
    
    RETURN
    pc_df:          data frame
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig = plt.figure('polcurve_comp')
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
        
        plt.scatter(current, voltage, label=v.name)
        
    fig.legend()
        
    if save_csv:
        from tkinter import filedialog
        from .paths import base_paths
        import os
        data_dir = base_paths()[1]
        data_dir = os.path.join(data_dir, '_polcurve_csv_export/')
        save_path = filedialog.asksaveasfilename(defaultextension='.csv', initialdir=data_dir)
        pc_df.to_csv(save_path, sep='\t')
        
    if show:
        plt.show()
    
    plt.close(fig=fig)
    
    return pc_df