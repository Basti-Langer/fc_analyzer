class eis_class:
    '''
    Class for electrochemical impedance spectroscopy (eis) data
    '''
    def __init__(self, path):
        import os
        import pandas as pd
        import numpy as np
        import time
        from .paths import mea_from_path
        from .constants import get_active_area
        
        self.path = path
        self.name = os.path.basename(path)
        
        self.mea = mea_from_path(path)
        
        date = os.path.getmtime(path)
        self.date = time.ctime(date)
        
        with open(path, 'r', encoding='cp1252') as f: 
            lines = f.readlines()
            for i in lines:
                if i.find('Time') != -1:
                    skiprows = lines.index(i) # anchor for the first of the two headers
                if i.find('FREQINIT') != -1:
                    f_init = i.split()
                    f_init = float(f_init[2])
                if i.find('FREQFINAL') != -1:
                    f_final = i.split()
                    f_final = float(f_final[2])
                if i.find('PTSPERDEC') != -1:
                    ppd = i.split()
                    ppd = float(ppd[2])
        
        self.f_range = (f_init, f_final)
        self.ppd = ppd
        data = pd.read_csv(self.path, sep='\t', 
                                encoding='cp1252', header=[skiprows, skiprows+1], dtype=np.float64)
        data = data.droplevel(level=1, axis=1)
        
        A = get_active_area()
        data['Z_complex'] = data['Zreal'] + data['Zimag']*1j
        data['Zreal [Ohmcm2]'] = data['Zreal'] * A # convert into Ohmcm2
        data['Zimag [Ohmcm2]'] = data['Zimag'] * A
        data['Z_complex [Ohmcm2]'] = data['Z_complex'] * A
        self.data = data       
    
    def get_hfr(self, mode: str='intersect', Zimag_range: list=[0.0015, 0.005], fit_points: float=5,  
                min_freq: float=100, max_freq: float=100e3, induct: bool=False, cycle=None,
                show: bool=False, save: bool=False):
        '''
        Determine the HFR of a specific eis spectrum
        
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
        
        'show' and 'save' (bool) refer to the plot
        
        'linreg' object is returned which has several properties as defined in scipy.stats.linregress
        
        'l' is the tuple of the inductivity (in H) and the beta exponent
        
        'cycle' specifies the cycle number of an AST (for possible lookup)
        
        INPUT
        mode:           str
        Zimag_range:    list
        fit_points:     float
        min_freq:       float
        max_freq:       float
        induct:         bool
        cycle:          float
        show:           bool
        save:           bool
        
        RETURN
        hfr:            float (in Ohm cm2)
        linreg:         linear regression object
        l:              tuple
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        from .lookup import lookup_xls
        from .eis_models import inductor
        from .constants import get_active_area
        
        data = self.data.copy()
        A = get_active_area()
        
        # Calculate inductivity
        # Zero inductivity for induct=False
        if induct == True:
            mea = self.mea
            # (L, beta); L in H
            l = (lookup_xls(mea, 'Inductivity L', cycle=cycle),  lookup_xls(mea, 'Inductivity beta', cycle=cycle))
        else:
            l = (0, 1) # L, beta
        ind = inductor(data['Freq'], l)
        
        # Update the raw data with the inductivity correction
        data['Z_complex'] = (data['Z_complex'] - ind)
        data['Z_complex [Ohmcm2]'] = data['Z_complex'] * A
        
        # select data for linear fit region
        lin_region = data.copy()
        cond_f_range =  ((lin_region['Freq'] >= min_freq) & (lin_region['Freq'] <= max_freq) & (lin_region['Zreal'] > 0))
        lin_region = lin_region.loc[cond_f_range, ['Zreal', 'Zreal [Ohmcm2]', 'Zimag', 'Zimag [Ohmcm2]', 
                                                   'Freq', 'Z_complex', 'Z_complex [Ohmcm2]']]
        
        if mode == 'linfit':
            # filter by Z_imag_range
            cond_Zimag_range = (-np.imag(lin_region['Z_complex [Ohmcm2]']) > Zimag_range[0]) & (-np.imag(lin_region['Z_complex [Ohmcm2]']) < Zimag_range[1])
            lin_region = lin_region.loc[cond_Zimag_range, ['Zreal', 'Zreal [Ohmcm2]', 'Zimag', 'Zimag [Ohmcm2]',
                                                           'Freq', 'Z_complex', 'Z_complex [Ohmcm2]']]

            # Minimal number of data points
            if len(lin_region) < 2: # at least three points
                raise ValueError('Less than 2 points in range. Increase Zimag_range')
            
        elif mode == 'intersect':
            # sort by modulus of Zimag and get the # points closest to the real axis (# = fit_points)
            lin_region = lin_region.sort_values(by='Z_complex', key=lambda x: np.abs(-np.imag(lin_region['Z_complex'])))
            lin_region = lin_region.iloc[:fit_points]
            
        else:
            raise ValueError('Select a known mode')
            
        # Apply linear regression
        linreg = stats.linregress(np.real(lin_region['Z_complex [Ohmcm2]']), -np.imag(lin_region['Z_complex [Ohmcm2]']))
        hfr = - linreg[1] / linreg[0] # x_0 = - c/m for y = m*x + c; in Ohm cm2
        hfr = float(hfr)
        
        # Check for inter- vs. extrapolation
        sign = min(np.imag(lin_region['Z_complex'])) * max(np.imag(lin_region['Z_complex']))
        if sign > 0:
            extrapol = True
        else:
            extrapol = False
        
        # plot
        fig = plt.figure(f'{mode}_HFR_{self.name}')
        plt.scatter(np.real(data['Z_complex [Ohmcm2]']), -np.imag(data['Z_complex [Ohmcm2]']), s=10, color='grey')
        plt.scatter(np.real(lin_region['Z_complex [Ohmcm2]']), -np.imag(lin_region['Z_complex [Ohmcm2]']), s=10, color='r')
        x_lim = [np.real(lin_region['Z_complex [Ohmcm2]']).min(), np.real(lin_region['Z_complex [Ohmcm2]']).max()]
        y_lim = [np.imag(-lin_region['Z_complex [Ohmcm2]']).min(), np.imag(-lin_region['Z_complex [Ohmcm2]']).max()]
        y1 = linreg[0] * x_lim[0] + linreg[1] # y = m*x + c
        y2 = linreg[0] * x_lim[1] + linreg[1] 
        plt.plot(x_lim, [y1, y2], color='r')
        plt.xlabel('$Z_{real} [\Omega \cdot cm^2]$')
        plt.ylabel('$-Z_{imag} [\Omega \cdot cm^2]$')
        plt.title(f'$HFR = {round((hfr*1000),2)} \, m\Omega cm^2$')
        plt.xlim(x_lim[0]-0.001, x_lim[1]+0.001)
        plt.ylim(y_lim[0]-0.001, y_lim[1]+0.001)
        plt.gca().set_aspect('equal')
            
        if save:
            import os
            save_path = f'{os.path.splitext(self.path)[0]}_HFR_{mode}_induct{str(induct)}.png'
            plt.savefig(save_path, format='png')
            
        if show:
            plt.show()
        
        plt.close(fig=fig)
        
        return hfr, linreg, l, extrapol
    
    def nyquist(self, save: bool=False, show: bool=True, raw: bool=False, induct: bool=False):
        '''
        Plot the eis data in a Nyquist representation
        
        'induct=True' allows to correct the raw data for the inductivity 
        
        'raw=True' plots the  impedance without area-normalization
        
        INPUT
        raw:        bool
        induct:     bool
        save:       bool
        show:       bool
        
        RETURN
        Z:          list of complex float
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        from .lookup import lookup_xls
        from .eis_models import inductor
        from .constants import get_active_area
        
        fig = plt.figure(f'Nyquist {self.name}')

        # correct the impedance (Z) in Ohm for the inductivity
        if induct:
            import numpy as np
            mea = self.mea
            l = (lookup_xls(mea, 'Inductivity L'),  lookup_xls(mea, 'Inductivity beta')) # (L, beta); L in H
            induct = inductor(self.data['Freq'], l) # in Ohm, complex number
            Z = (self.data['Z_complex'] - induct)
        else:
            Z = self.data['Z_complex']
        
        # display as NON-normalized impedance (as measured)
        if raw:
            plt.xlabel('$Z_{real} \,[\Omega]$')
            plt.ylabel('$-Z_{imag} \,[\Omega]$')
        # display as area-normalized impedance    
        else:
            A = get_active_area()
            Z = Z * A  
            plt.xlabel('$Z_{real} \, [\Omega \cdot cm^2]$')
            plt.ylabel('$-Z_{imag} \, [\Omega \cdot cm^2]$')
        
        # plot   
        plt.scatter(np.real(Z), -np.imag(Z), marker='s', s=12, color='k')
        plt.gca().set_aspect('equal')

        if save:
            import os
            save_path = f'{os.path.splitext(self.path)[0]}_nyquist.png'
            plt.savefig(save_path, format='png')
        
        if show:
            plt.show()
            
        plt.close(fig=fig)
            
        return np.real(Z), np.imag(Z)
    
    def bode(self, save: bool=False, show: bool=True, hfr_corr: bool=True, induct: bool=False):
        '''
        Plot the eis data in a Bode representation (phase angle vs. log(frequency))
        
        'hfr_corr=True' reads the HFR from local file and corrects the data for this HFR
        
        'induct=True' allows to correct the raw data for the inductivity 

        INPUT
        hfr_corr:   bool
        induct:     bool
        save:       bool
        show:       bool
        
        RETURN
        freq:       list
        phi:        list
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        from .eis_models import inductor
        from .lookup import lookup_xls
        from .find_open import read_hfrs
        
        fig = plt.figure(f'Bode {self.name}')
        plt.xlabel(r'$f [Hz]$')
        plt.ylabel(r'$\Phi [°]$')
        plt.xscale('log')
        freq = self.data['Freq']
        
        if induct:
            mea = self.mea
            l = (lookup_xls(mea, 'Inductivity L'),  lookup_xls(mea, 'Inductivity beta')) # (L, beta); L in H
            induct = inductor(self.data['Freq'], l) # in Ohm, complex number
            Z = (self.data['Z_complex'] - induct)
        else:
            Z = self.data['Z_complex']
        
        # correct for HFR as saved in local file
        if hfr_corr:
            HFRs = read_hfrs(os.path.dirname(self.path))
            try:
                hfr = HFRs.loc[self.name, 'HFR [Ohmcm2]']
            except:
                raise ValueError(f'no HFR found for {self.name}')
        
            Z = Z - hfr
            
        phi = - np.angle(Z, deg=True)
        
        plt.scatter(freq, phi, marker='s', s=10, label=self.name)
        fig.legend()

        if save:
            import os
            save_path = f'{os.path.splitext(self.path)[0]}_bode.png'
            plt.savefig(save_path, format='png')
            
        if show:
            plt.show()
            
        plt.close(fig=fig)
            
        return freq, phi
        
    def fit_induct(self, freq_range: list=[30e3, 100e3], p0: list=[1e-2, 1e-8, 1], show: bool=True, save: bool=False):
        '''
        Fit the eis data to a modified inductor with a real resistor in series

        Circuit: R0-Lmod0

        freq_range=[a, b] defines the lower (a) and upper (b) limit of considered frequencies

        'p0' are the starting parameters for the fit in given units: $HFR (\Omega), L (H), \beta ( )$

        'show' and 'save' refer to the plot and to saving the fit parameters into a local 'fit_summary.csv' file

        INPUT
        freq_range:      list
        p0:             list
        save:           bool
        show:           bool

        RETURN
        R:              float
        L:              float
        beta:           float
        '''
        from impedance.models.circuits import CustomCircuit
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from .constants import get_active_area
        
        # Define model
        circuit = 'R0-Lmod0'
        model = CustomCircuit(circuit=circuit, initial_guess=p0)
        
        # select data
        data = self.data.copy()
        cond = (data['Freq'] > freq_range[0]) & (data['Freq'] < freq_range[1])
        data = data.loc[cond, ['Freq', 'Z_complex']]
        f = data['Freq']
        Z = data['Z_complex']
        
        # fit
        model.fit(f, Z)
        fit = model.predict(f)
        
        # get fitting parameters
        R, L, beta = model.parameters_
        R = R * get_active_area() # in Ohm cm2

        # plot
        fig = plt.figure(f'Inductivity fit {self.name}')
        plt.gca().set_aspect('equal')
        plt.xlabel('$Z_{real} \, [\Omega]$')
        plt.ylabel('$-Z_{imag} \, [\Omega]$')
        plt.scatter(np.real(Z), -np.imag(Z), color='k', label='data')
        plt.plot(np.real(fit), -np.imag(fit), color='r')
        plt.scatter(np.real(fit), -np.imag(fit), marker='o', facecolor='none', edgecolor='r', label='fit')
        plt.title(r'$HFR = $' + str(round(R*1000, 3)) + r'$\, m\Omega \cdot cm^2 \, , L = $' + str(round(L*1e9,3)) + r'$\,nH, \, \beta =$' + str(round(beta,3)))
        fig.legend()
        
        if save:
            import os
            import pandas as pd
            from datetime import datetime
            from .util import fit_summary_df
            
            dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            
            # save figure
            save_path = f'{os.path.splitext(self.path)[0]}_induct-fit.png'
            plt.savefig(save_path, format='png')
            
            # save fitting results in local file
            fit_result = fit_summary_df()
            
            fit_result.at[0, 'name'] = self.name
            fit_result.at[0, 'date of fit'] = dt
            fit_result.at[0, 'model'] = circuit
            fit_result.at[0, 'freq_range [Hz]'] = freq_range
            fit_result.at[0, 'HFR [Ohmcm2]'] = R
            fit_result.at[0, 'HFR_conf'] = model.conf_[0]
            fit_result.at[0, 'L [H]'] = L
            fit_result.at[0, 'L_conf'] = model.conf_[1]
            fit_result.at[0, 'beta'] = beta
            fit_result.at[0, 'beta_conf'] = model.conf_[2]
            
            save_path = os.path.join(os.path.dirname(self.path), 'eis_fit_summary.csv')
            
            # check if there is already a summary file
            if os.path.isfile(save_path):
                fit_result_old = pd.read_csv(save_path, sep= '\t')
                fit_result = pd.concat([fit_result_old, fit_result], ignore_index=True)
            
            fit_result.to_csv(save_path, sep='\t', index=False)
            
        if show:
            plt.show()
            
        return R, L, beta
    
    import numpy as np
    def fit_tlm_blocking(self, p0: list=[1e-2,1e-2,1e-1,1,1e-8,1],
                         bounds=([0, 0, 0, 0, 1e-10, 0], [np.inf, np.inf, np.inf, 1, 1e-7, 1]),
                         freq_range: list=[1, 1e5], cycle: int=None, show: bool=True,
                         save: bool=False, induct: bool=True, global_opt: bool=False):
        '''
        Fit impedance spectrum under blocking conditions
        
        Circuit:    R0 - TLMQ - La
        
        p0: Starting fit parameters
        ['R0', 'TLMQ0_0', 'TLMQ0_1', 'TLMQ0_2', 'La0_0', 'La0_1'],
        in units of:
        ['Ohm', 'Ohm', 'F sec^(gamma - 1)', '', 'H sec', '']
        
        'bounds' gives the boundaries for each fitting parameter
        as a tuple of array-like lower bounds and upper bounds
        
        freq_range=[a, b]defines the lower (a) and upper (b) limit of considered frequencies
        
        'induct=True' sets the inductivity constant
        
        'global_opt=True' changes fit algorithm to global optimization (scipy basinhopping).
        Mind longer computation time
        
        'induct=True' read inductivity from Excel and set inductivity constant
        
        'show' and 'save' refer to the plot and to saving the fit parameters into a local 'fit_summary.csv' file
        
        INPUT
        freq_range:     list
        p0:             list
        save:           bool
        show:           bool
        induct:         bool
        global_opt:     bool

        RETURN
        R:              float
        Rp:             float
        Q:              float
        alpha:          float
        L:              float
        beta:           float
        '''
        from impedance.models.circuits import CustomCircuit
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from .lookup import lookup_xls
        from .constants import get_active_area
        
        # define model
        circuit = 'R0-TLMQ0-Lmod0'
        if induct:
            mea = self.mea
            # (L, beta); L in H
            l = (lookup_xls(mea, 'Inductivity L', cycle=cycle),  lookup_xls(mea, 'Inductivity beta', cycle=cycle))
            L = l[0]
            beta = l[1]
            constants = {'Lmod0_0': L, 'Lmod0_1': beta}
            # if inductivity is set constant, delete from starting parameters and bounds
            p0 = p0[:-2]
            bounds = (bounds[0][:-2], bounds[1][:-2])
        else:
            constants={}
        model = CustomCircuit(circuit=circuit, initial_guess=p0, constants=constants)
        
        # select data
        data = self.data.copy()
        cond = (data['Freq'] > freq_range[0]) & (data['Freq'] < freq_range[1])
        data = data.loc[cond, ['Freq', 'Z_complex']]
        f = data['Freq']
        Z = data['Z_complex']
        
        # fit
        model.fit(f, Z, global_opt=global_opt, bounds=bounds)
        fit = model.predict(f)
        
        # get fitting parameters
        if induct:
            R, Rp, Q, alpha = model.parameters_
        else:
            R, Rp, Q, alpha, L, beta = model.parameters_
        
        # Adapt units    
        A = get_active_area()
        R = R * A # in Ohm cm2
        Rp = Rp * A # in Ohm cm2
        Q = Q / A # in F/cm2
        Z = Z * A # in Ohm cm2
        fit = fit * A # in Ohm cm2

        # plot
        fig = plt.figure(f'TLM-blocking fit {self.name}')
        plt.gca().set_aspect('equal')
        plt.xlabel('$Z_{real} \, [\Omega \cdot cm^2]$')
        plt.ylabel('$-Z_{imag} \, [\Omega \cdot cm^2]$')
        plt.scatter(np.real(Z), -np.imag(Z), color='k', label='data')
        plt.plot(np.real(fit), -np.imag(fit), color='r')
        plt.scatter(np.real(fit), -np.imag(fit), marker='o', facecolor='none', edgecolor='r', label='fit')
        plt.title(r'$HFR = $' + str(round(R*1000, 3)) + r'$\, m \Omega \cdot cm^2 \, , \, R_{p} = $' + str(round(Rp*1000, 3)) + r'$\, m \Omega \cdot cm^2 , \, Q =$' + str(round(Q*1000,3)) + r'$\,mF/cm^2$')
        fig.legend()
        
        if save:
            import os
            import pandas as pd
            from datetime import datetime
            from .util import fit_summary_df
            
            dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            
            # save figure
            save_path = f'{os.path.splitext(self.path)[0]}_tlm-fit.png'
            plt.savefig(save_path, format='png')
            
            # save fitting results in local file
            fit_result = fit_summary_df()
            
            fit_result.at[0, 'name'] = self.name
            fit_result.at[0, 'date of fit'] = dt
            fit_result.at[0, 'model'] = circuit
            fit_result.at[0, 'induct_mode'] = str(induct)
            fit_result.at[0, 'freq_range [Hz]'] = freq_range
            fit_result.at[0, 'HFR [Ohmcm2]'] = R
            fit_result.at[0, 'HFR_conf'] = model.conf_[0]
            fit_result.at[0, 'R_cath [Ohmcm2]'] = Rp
            fit_result.at[0, 'R_cath_conf'] = model.conf_[1]
            fit_result.at[0, 'Q [F/cm2]'] = Q
            fit_result.at[0, 'Q_conf'] = model.conf_[2]
            fit_result.at[0, 'alpha'] = alpha
            fit_result.at[0, 'alpha_conf'] = model.conf_[3] 
            fit_result.at[0, 'L [H]'] = L
            fit_result.at[0, 'beta'] = beta
            if induct == False:
                fit_result.at[0, 'L_conf'] = model.conf_[4]
                fit_result.at[0, 'beta_conf'] = model.conf_[5] 
            
            save_path = os.path.join(os.path.dirname(self.path), 'eis_fit_summary.csv')
            
            # check if there is already a summary file
            if os.path.isfile(save_path):
                fit_result_old = pd.read_csv(save_path, sep= '\t')
                fit_result = pd.concat([fit_result_old, fit_result], ignore_index=True)
            
            fit_result.to_csv(save_path, sep='\t', index=False)
                
        if show:
            plt.show()
            
        plt.close(fig=fig)
        print("R(HFR)[Ohmcm2], Rp(cathode)[Ohmcm2], Q, alpha, L, beta")
        return R, Rp, Q, alpha, L, beta
    
    def fit_tlm_load(self, p0: list=[1e-2, 1e-2, 1e-1, 1, 0.1, 1e-8, 1],
                     bounds=([0, 0, 0, 0, 0, 1e-10, 0], [np.inf, np.inf, np.inf, 1, np.inf, 1e-7, 1]),
                     freq_range: list=[1, 1e5], cycle: int=None, show: bool=True, save: bool=False,
                     induct: bool=True, global_opt: bool=False):
        '''
        Fit eis spectrum under load (i.e. with charge transfer resistance)
        
        Circuit:
        R0 - TLMRct - Lmod
        
        p0 starting fit parameters
        ['R0', 'R_cath', 'Q', 'alpha', 'Rct', 'La0_0', 'La0_1'],
        in units of
        ['Ohm', 'Ohm', 'F sec^(gamma - 1)', '', 'Ohm', 'H sec', '']
        
        freq_range=[a, b]defines the lower (a) and upper (b) limit of considered frequencies
        
        'induct=True' sets the inductivity constant
        
        'global_opt=True' changes fit algorithm to global optimization (scipy basinhopping).
        Mind longer computation time
        
        'induct=True' read inductivity from Excel and set inductivity constant
        
        'show' and 'save' refer to the plot and to saving the fit parameters into a local 'fit_summary.csv' file
        
        INPUT
        freq_range:     list
        p0:             list
        save:           bool
        show:           bool
        induct:         bool
        global_opt:     bool

        RETURN
        R:              float
        Rp:             float
        Q:              float
        alpha:          float
        Rct:            float
        L:              float
        beta:           float
        '''
        from impedance.models.circuits import CustomCircuit
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from .lookup import lookup_xls
        from .constants import get_active_area
        
        # define model
        circuit = 'R0-TLMRct0-Lmod0'
        if induct:
            mea = self.mea
            # (L, beta); L in H
            l = (lookup_xls(mea, 'Inductivity L', cycle=cycle),  lookup_xls(mea, 'Inductivity beta', cycle=cycle))
            L = l[0]
            beta = l[1]
            constants = {'Lmod0_0': L, 'Lmod0_1': beta}
            # if inductivity is set constant, delete from starting parameters and bounds
            p0 = p0[:-2]
            bounds = (bounds[0][:-2], bounds[1][:-2])
        else:
            constants={}
        model = CustomCircuit(circuit=circuit, initial_guess=p0, constants=constants)
        
        # select data
        data = self.data.copy()
        cond = (data['Freq'] > freq_range[0]) & (data['Freq'] < freq_range[1])
        data = data.loc[cond, ['Freq', 'Z_complex']]
        f = data['Freq']
        Z = data['Z_complex']
        
        # fit
        model.fit(f, Z, global_opt=global_opt, bounds=bounds)
        fit = model.predict(f)
        
        # get fitting parameters
        if induct:
            R, Rp, Q, alpha, Rct = model.parameters_
        else:
            R, Rp, Q, alpha, Rct, L, beta = model.parameters_
        
        # Adapt units
        A = get_active_area()
        R = R * A # in Ohm cm2
        Rp = Rp * A # in Ohm cm2
        Q = Q / A # in F/cm2
        Rct = Rct * A # in Ohm cm2
        Z = Z * A # in Ohm cm2
        fit = fit * A # in Ohm cm2

        # plot
        fig = plt.figure(f'TLM-load fit {self.name}')
        plt.gca().set_aspect('equal')
        plt.xlabel('$Z_{real} \, [\Omega \cdot cm^2]$')
        plt.ylabel('$-Z_{imag} \, [\Omega \cdot cm^2]$')
        plt.scatter(np.real(Z), -np.imag(Z), color='k', label='data')
        plt.plot(np.real(fit), -np.imag(fit), color='r')
        plt.scatter(np.real(fit), -np.imag(fit), marker='o', facecolor='none', edgecolor='r', label='fit')
        plt.title(r'$HFR = $' + str(round(R*1000, 3)) + r'$\, m\Omega \cdot cm^2 \, , \, R_{p} = $' + str(round(Rp*1000, 3)) + r'$\,m\Omega \cdot cm^2 , \, Q =$' + str(round(Q*1000,3)) + r'$\,mF/cm^2, \, R_{ct}=$' + str(round(Rct*1000, 3)) + r'$\, m\Omega \cdot cm^2$')
        fig.legend()
        
        if save:
            import os
            import pandas as pd
            from datetime import datetime
            from .util import fit_summary_df
            
            dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            
            # save figure
            save_path = f'{os.path.splitext(self.path)[0]}_tlm-fit.png'
            plt.savefig(save_path, format='png')
            
            # save fitting results in local file
            fit_result = fit_summary_df()
            
            fit_result.at[0, 'name'] = self.name
            fit_result.at[0, 'date of fit'] = dt
            fit_result.at[0, 'model'] = circuit
            fit_result.at[0, 'induct_mode'] = str(induct)
            fit_result.at[0, 'freq_range [Hz]'] = freq_range
            fit_result.at[0, 'HFR [Ohmcm2]'] = R
            fit_result.at[0, 'HFR_conf'] = model.conf_[0]
            fit_result.at[0, 'R_cath [Ohmcm2]'] = Rp
            fit_result.at[0, 'R_cath_conf'] = model.conf_[1]
            fit_result.at[0, 'Q [F/cm2]'] = Q
            fit_result.at[0, 'Q_conf'] = model.conf_[2]
            fit_result.at[0, 'alpha'] = alpha
            fit_result.at[0, 'alpha_conf'] = model.conf_[3]
            fit_result.at[0, 'R_ct [Ohmcm2]'] = Rct
            fit_result.at[0, 'R_ct_conf'] = model.conf_[4]
            fit_result.at[0, 'L [H]'] = L
            fit_result.at[0, 'beta'] = beta
            if induct == False:
                fit_result.at[0, 'L_conf'] = model.conf_[5]
                fit_result.at[0, 'beta_conf'] = model.conf_[6]
            
            save_path = os.path.join(os.path.dirname(self.path), 'eis_fit_summary.csv')
            
            # check if there is already a summary file
            if os.path.isfile(save_path):
                fit_result_old = pd.read_csv(save_path, sep= '\t')
                fit_result = pd.concat([fit_result_old, fit_result], ignore_index=True)
            
            fit_result.to_csv(save_path, sep='\t', index=False)
                
        if show:
            plt.show()
            
        plt.close(fig=fig)
            
        return R, Rp, Q, alpha, Rct, L, beta
   