def window_geometry(ws, hs):
    '''
    Defines the geometry of a tkinter window
    '''
    # Select width and height of window
    w = 4000
    h = 4000
    
    # Calculcate width and height of window
    x = int((ws/2) - (w/2))
    y = int((hs/2) - (h/2))
    
    return w, h, x, y

def select_files():
    '''
    Opens the file explorer to select multiple files
    '''
    from .paths import base_paths
    from tkinter import Tk
    from tkinter import filedialog
    
    root = Tk()
    root.withdraw()
    data_dir = base_paths()[1]
    
    # Width and height of screen
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    
    w, h, x, y = window_geometry(ws, hs)
    root.geometry(f'{w}x{h}+{x}+{y}')
    
    paths = filedialog.askopenfilenames(initialdir=data_dir)
    paths = list(paths)
    
    root.destroy()
    
    return paths

def select_folder():
    '''
    Opens the file explorer to select a directory
    '''
    from tkinter import Tk
    from tkinter import filedialog
    from .paths import base_paths
    
    root = Tk()
    root.withdraw()
    data_dir = base_paths()[1]
    
    # Width and height of screen
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    
    w, h, x, y = window_geometry(ws, hs)
    root.geometry(f'{w}x{h}+{x}+{y}')
    
    dir0 = filedialog.askdirectory(initialdir=data_dir) # polcurve directory
    root.destroy()
    
    print(f'Selected directory: {dir0}')
    
    return dir0

def fit_summary_df():
    '''
    Initialize a generalized data frame for saving eis fitting results
    '''
    import pandas as pd
    
    fit_result = pd.DataFrame(columns=['name', 'date of fit', 'model',
                                       'induct_mode', 'freq_range [Hz]', 
                                       'HFR [Ohmcm2]', 'HFR_conf',
                                       'R_cath [Ohmcm2]', 'R_cath_conf',
                                       'Q [F/cm2]', 'Q_conf',
                                       'alpha', 'alpha_conf',
                                        'R_ct [Ohmcm2]', 'R_ct_conf',
                                        'L [H]', 'L_conf', 
                                        'beta', 'beta_conf',
                                        'sigma0 [S/cm2]', 'sigma0_conf'
                                        'gamma', 'gamma_conf'])
    return fit_result