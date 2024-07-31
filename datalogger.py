class data_logger_class:
    '''
    Class for log files (.csv) or tdms files
    Both are read-in as pandas data frame
    '''
    def __init__(self, path):
        import os
        import pandas as pd
        import time
        from nptdms import TdmsFile
        from .paths import mea_from_path
        
        self.path = path
        self.name = os.path.basename(path)
        date = os.path.getmtime(path)
        self.date = time.ctime(date)
        
        self.mea = mea_from_path(path)
        
        # open file depending on file extension
        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            with open(path, 'r', encoding='cp1252') as f: 
                lines = f.readlines()
                for i in lines:
                    if i.find('Time Stamp') != -1:
                        skiprows = lines.index(i)
                    elif 'G19-6710' in i:
                        station = 5
                    elif 'G19-6712' in i:
                        station = 6
            self.station = station
            
            self.data = pd.read_csv(path, sep=',', encoding='cp1252', skiprows=skiprows)

        elif ext == '.tdms':
            file = TdmsFile.read(path)
            self.data = file['Group1'].as_dataframe()
            
    def e_i(self):
        '''
        Plot the cell potential and the current vs. time
        
        INPUT
        None
        
        RETURN
        time:       list
        current:    list
        voltage:    list
        '''
        import matplotlib.pyplot as plt
        
        # get data
        data = self.data
        time = data['Elapsed Time']
        current = data['current_density']
        voltage = data['cell_voltage_001']

        # plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylabel(r'$E\, [V]$')
        ax2.set_ylabel(r'$i\, [A/cm2]$')
        ax1.set_xlabel(r'$t\, [s]$')
        
        ax1.plot(time, voltage, color='b', label='cell voltage')
        ax2.plot(time, current, color='r', label='load current')
        
        fig.legend()
        plt.show()
        
        return time, current, voltage
    
    def temps(self):
        '''
        Plot the cell potential and temperatures (cell and dew point) vs. time
        
        INPUT
        None
        
        RETURN
        time:       list
        voltage:    list
        T_cell_a:   list
        T_cell_c:   list
        T_dew_a:    list
        T_dew_c:    list
        '''
        import matplotlib.pyplot as plt
        
        # get data
        data = self.data
        time = data['Elapsed Time']
        voltage = data['cell_voltage_001']
        T_cell_a = data['temp_anode_endplate']
        T_cell_c = data['temp_cathode_endplate']
        T_dew_a = data['temp_anode_dewpoint_water']
        T_dew_c = data['temp_cathode_dewpoint_water']
        
        # plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylabel(r'$E\, [V]$')
        ax2.set_ylabel(r'$T\, [°C]$')
        ax1.set_xlabel(r'$t\, [s]$')
        
        ax1.plot(time, voltage, color='k', label='voltage')
        ax2.plot(time, T_cell_a, label='T_cell_a')
        ax2.plot(time, T_cell_c, label='T_cell_c')
        ax2.plot(time, T_dew_a, label='T_dew_a')
        ax2.plot(time, T_dew_c, label='T_dew_c')
        
        fig.legend()
        plt.show()
        
        return time, voltage, T_cell_a, T_cell_c, T_dew_a, T_dew_c
        
    def gamry(self):
        '''
        Plot the cell potential, cell current, Gamry potential, and Gamry current
        
        INPUT
        None
        
        RETURN
        time:       list
        voltage:    list
        current:    list
        gamry_e:    list
        gamry_i:    list
        '''
        import matplotlib.pyplot as plt
        
        # get data
        data = self.data
        time = data['Elapsed Time']
        voltage = data['cell_voltage_001']
        current = data['current_density']
        gamry_e = data['Gamry-01.voltage']
        gamry_i = data['Gamry-01.current']
        
        # plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_ylabel(r'$E\, [V]$')
        ax2.set_ylabel(r'$i\, [A/cm2]$')
        ax1.set_xlabel(r'$t\, [s]$')
        
        ax1.plot(time, voltage, color='k', label='cell voltage')
        ax1.plot(time, gamry_e, color='grey', label='Gamry voltage')
        ax2.plot(time, current, label='load current')
        ax2.plot(time, gamry_i, label='Gamry current')
        
        fig.legend()
        plt.show()
        
        return time, voltage, current, gamry_e, gamry_i

        