methods = ['polcurve', 'eis', 'cv', 'log', 'chrono', 'ilim'] # for each method one class (i.e. data type)

def read_data(paths: list, method: str):
    from .polcurve import polcurve_class
    from .eis import eis_class
    from .cv import cv_class
    from .datalogger import data_logger_class
    from .chrono import chrono_class
    from .ilim import ilim_class
    '''
    Reads paths (list) and initializes the files in the respective class as object
    All paths given must correspond to the same method
    
    INPUT
    paths:      list of str
    method:     str
    
    RETURN
    data:       list of objects of the respective class
    '''
    import pandas as pd

    if len(paths) == 0:
        raise Exception('no data')
    
    data = []

    for i in paths:
        if method == 'polcurve':
            pc = polcurve_class(i)
            data.append(pc)
        
        elif method == 'eis':
            eis = eis_class(i)
            data.append(eis)
        
        elif method == 'cv':
            cv = cv_class(i)
            data.append(cv)
        
        elif method == 'log':
            log = data_logger_class(i)
            data.append(log)
        
        elif method == 'chrono':
            chrono = chrono_class(i)
            data.append(chrono)
            
        elif method == 'ilim':
            ilim = ilim_class(i)
            data.append(ilim)
    
    return data

def get_paths(mode: str='explore'):
    '''
    "FC data search engine" with two possible modes:
    - 'explore':    select data in the file explorer
    - 'filter' :    select data by applying a filter to the data base

    Uses terminal input.

    Filter method uses recursive search

    Empty input means not to filter by this argument

    Search criteria within one method are exclusive (i.e. criteria must be met)

    Archive folder is always excluded

    'keyword' and 'exclude' are useful to filter by substrings of the filename

    INPUT
    mode:       str

    RETURN
    paths:      list of str
    method:     method given as terminal input
    '''
    import os
    import glob
    from .paths import base_paths
    
    gases = ['O2', 'air']
    rhs = ['30', '50', '70', '90', '95', '100']
    
    method = input('Method:\t')
    method = method.lower()
    while method not in methods:
        print(f'Choose method from \n {methods}')
        method = input('Method:\t')
    
    # select files using the file explorer   
    if mode == 'explore':
        from .util import select_files

        paths = select_files()
    
    # Filter files by terminal input
    if mode == 'filter':
        paths = [] # list of relevant paths that are returned
        dir0 = base_paths()[1]
        
        mea = input('MEA # (comma separated):\t')
        mea = mea.split(',')
        mea = [m.strip() for m in mea] # list of strings

        # several MEAs can be selected simulataneously
        for m in mea:
            mea_dir = os.path.join(dir0, f'B-MEA-{m}')

            # only one method is selected
            # Polcurves
            if method == 'polcurve':
                data_dir = os.path.join(mea_dir, 'polcurves')
                
                pc_paths = glob.glob(os.path.join(data_dir, '**/*PolCurve*.csv'), recursive=True) # all polcurves

                gas = input(f'Cathode gas B-MEA-{m}:\t')
                while (len(gas) > 0) & (gas not in gases):
                    print(f'Choose gas from \n {gases}')
                    gas = input(f'Cathode gas B-MEA-{m}:\t')
                    
                pc_paths = [p for p in pc_paths if f'H2{gas}' in p]        
                
                rh = input(f'RH (%) B-MEA-{m}:\t')
                while (len(rh) > 0) & (rh not in rhs):
                    print(f'Choose RH from \n {rhs}')
                    gas = input(f'RH (%) B-MEA-{m}:\t')                
                try:
                    rh = float(rh)/100 # float of empty string is error -> except
                    pc_paths = [p for p in pc_paths if f'{rh:.2f}RH' in p] 
                except:
                    pass
            
                paths = paths + pc_paths
            
            # EIS files 
            elif method == 'eis':
                # subdirectory is not specified here
                eis_paths = glob.glob(os.path.join(mea_dir, '**/*z data*.DTA'), recursive=True)
                paths = paths + eis_paths
                
            # CV
            elif method == 'cv':
                data_dir = os.path.join(mea_dir, 'CV')
                cv_paths = glob.glob(os.path.join(data_dir, '**/*cv data*.DTA'), recursive=True)
                paths = paths + cv_paths
            
            # data logger
            elif method == 'log':
                # subdirectory is not specified here
                log_paths = glob.glob(os.path.join(mea_dir, '**/*log*.csv'), recursive=True)
                paths = paths + log_paths
            
            # chrono
            elif method == 'chrono':
                chrono_paths = glob.glob(os.path.join(mea_dir, '**/*chrono data*.DTA'), recursive=True)
                paths = paths + chrono_paths
                
            # ilim
            elif method == 'ilim':
                data_dir = os.path.join(mea_dir, 'ilim')
                ilim_paths = glob.glob(os.path.join(data_dir, '**/*lim*.csv'), recursive=True)
                paths = paths + ilim_paths
   
        # keep only if keyword is found in path (not case sensitive)
        keyword = input('keyword (comma separated):\t')
        keyword = keyword.split(',')
        keyword = [k.strip() for k in keyword] # list of strings
        for k in keyword:
            if len(k) > 0:
                paths = [p for p in paths if k.lower() in p.lower()]

        # exclude path is 'exclude' is found (not case sensitive)
        exclude = input('exclude (comma separated):\t')
        exclude = exclude.split(',')
        exclude = [e.strip() for e in exclude]
        for e in exclude:
            if len(e) > 0:
                paths = [p for p in paths if e.lower() not in p.lower()]
                    
        # exclude 'archive' folder
        for i in paths:
            if i.find('archive') != -1:
                paths.remove(i)
                    
        # delete duplicates in paths
        paths = list(set(paths))
    
    # order by file creation time
    paths.sort(key=lambda p: os.path.getmtime(p))
        
    if len(paths) == 0:
        raise KeyError('no data found')
    else:
        for i,v in enumerate(paths):
            print(f'{i}: \t {os.path.split(v)[1]}')
    
    return paths, method

def get_data(mode: str='explore'):
    '''
    "FC data search engine" with two possible modes: 'explore' or 'filter'
    
    Combined method to get paths and then load the data as objects of the respective class
    
    Filter method uses recurisve search

    Empty input means not to filter by this argument

    Search criteria within one method are exclusive (i.e. criteria must be met)

    Archive folder is always excluded

    'keyword' and 'exclude' are useful to filter by substrings of the filename

    They are not case-sensitive and refer to the absolute path
    
    INPUT
    mode:       str
        'explore':  select data in the file explorer
        'filter' :  select data by applying a filter to the data base 
        
    RETURN
    data:       list of objects
    '''
    paths, method = get_paths(mode=mode)
    
    data = read_data(paths=paths, method=method)
    
    return data

def read_hfrs(dir):
    '''
    Reads the HFRs from summary file which must be in the directory given as input

    Option to select file, if various HFR files are found
    
    INPUT
    dir:           str
    
    RETURN
    HFRs            list
    '''
    import glob
    import pandas as pd
    import os
    
    HFR_paths = glob.glob(os.path.join(dir, '*HFR*.csv'))

    if len(HFR_paths) > 1:
        print('Select HFR file by index')
        for i, v in enumerate(HFR_paths):
            print(f'{i}:\t{v}')
            
        idx = input('Index:\t')
        HFR_path = HFR_paths[int(idx)]
    elif len(HFR_paths) == 1:
        HFR_path = HFR_paths[0]
    elif len(HFR_paths) == 0:
        raise ValueError('No HFR data found')
    
    HFRs = pd.read_csv(HFR_path, sep='\t', index_col=0)
    
    return HFRs # data frame
