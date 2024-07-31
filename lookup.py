def lookup_xls(mea, col, cycle=None):
    '''
    Looks up values from the overview Excel
    
    'mea' is the number of the MEA (row in Excel)
    
    'col' is the column to look up the value from in the Excel
    
    'cycle' is the number of voltage cycles in an AST
    
    INPUT
    mea:        int
    col:        str
    cycle:      float
    
    RETURN
    value:      float
    '''        
    from .paths import base_paths
    import pandas as pd
    import numpy as np
    
    xls_path, _, __, pattern  = base_paths()
    
    
#    try:
    avg = pd.read_excel(xls_path, sheet_name='Characteristics', header=1, index_col='MEA')
    value = avg.loc[f'{pattern}{mea}', col] # mind unit
    print("value")
    print(value)
#    except:
#        print(f'No data found for MEA {mea}')
#        value = input(f'Type {col} of MEA {mea}:\t')
#        value = float(value)
    
    # Only one single value found
    if type(value) == float:
        pass
    
    # More than one value is found
    elif isinstance(value, pd.Series):
        cycles = avg.loc[f'{pattern}{mea}', 'Cycle #']
        param = avg.loc[f'{pattern}{mea}', ('Cycle #', col)]
        
        if isinstance(cycle, int): # cycle is either None or int
            cycle_sel = cycle
        else:
            cycle_sel = input(f'\nMore than one value found for {col} of {pattern}{mea}.'
                            f'\nSelect cycle # from... \n {param} \n cycle =\t'
                            )
            cycle_sel = float(cycle_sel) # string to float
        
        while cycle_sel not in list(cycles):
            print(f'\nSelect cycle from... \n {list(cycles)}')
            cycle_sel = float(input('cycle =\t'))
        
        value = avg.loc[f'{pattern}{mea}', col][avg.loc[f'{pattern}{mea}', 'Cycle #'] == cycle_sel]
        value = float(value)
        
    elif (type(value) == str) | (np.isnan(value)):
        # strings and nan are not accepted
        print(f'No {col} data available for MEA {mea}')
        value = input(f'Type {col} of MEA {mea}:\t')
        value = float(value)
        
    return value

def lookup_cal():
    '''
    Load calibration data
    
    INPUT
    None
    
    RETURN
    cal:        data frame with calibration data
    methods:    calibration methods (headers of data frame)
    '''
    import pandas as pd
    from .paths import base_paths
    
    cal_path = base_paths()[2]
    cal = pd.read_csv(cal_path, sep=';')
    methods = cal.columns
    methods = methods[2:]
    
    return cal, methods