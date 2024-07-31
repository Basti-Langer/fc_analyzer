def base_paths():
    '''
    Get important paths in the personal directories.
    Should be checked if a different mounting is used
    '''
    import json
    from importlib.resources import files
    
    with files('fc_analyzer').joinpath('customize.json').open('r') as f:
        custom = json.load(f)
    
    xls_path = custom['xls_path']
    data_dir = custom['data_dir']
    cal_path = custom['cal_path']
    pattern = custom['MEA_pattern']
    
    return xls_path, data_dir, cal_path, pattern

def mea_from_path(path):
    '''
    Get the MEA number from a given path
    The function checks for a pattern like 'B-MEA-xx' to extract the number 'xx'
    
    INPUT
    path:       str
    
    RETURN
    mea:        int
    '''
    import re
    import json
        
    pattern = base_paths()[3]
    
    try:
        mea = re.search(fr'{pattern}\d*', str(path)) # takes the first occurence (e.g. directory)
        mea = mea.group()
        #mea = mea.split('-')[2]
        mea = re.split(r'[_C]+', mea)[1]
        mea = int(mea)
    except:
        import os
        mea = input(f'MEA # for {os.path.basename(path)}:\t')
        mea = int(mea)
        
    return mea

def rh_from_filename(path):
    '''
    Get the RH of given CV-File for RH-dependend CO-Strip
    Only filename is checked (independent of input is filename or path)

    INPUT
    filename: str

    RETURN
    rh: int
    '''
    import re
    import os

    rh = os.path.basename(path)
    
    try:
        rh = re.search(r'\d\.\d*RH', str(rh))
        rh = rh.group()
        rh = rh[:-2] # exclude the 'RH'
        rh = float(rh)
    except:
        rh = input(f'RH for {os.path.basename(path)}:\t')
        rh = int(rh)
    return rh