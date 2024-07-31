# Checkout how much Co2+ is in the cell

def cation_cal(val: float, method: str='HFR(30)', x_range: list=[0, 0.35],
               show: bool=True, fit_order: int=5, mode: str='poly'):
    '''
    Calculate precentage of exchnaged H+ according to interpolation
    of the calibration data
    
    Calibration data loaded from:
    '/mnt/pc4/Bosch/02_fuel_cell_data/02_data_analysis/04_WP4-model_CCMs_insitu_EIS/calibration_data/HFR30_Rho50_cal.csv'
    
    'val' is the input value corresonding to the method
    (for hfr30, 'val' is in mOhmcm2)
    
    'method' is the column nam of the calibration csv
    
    'fit_order' sets the order of the polynomial fit
    
    'mode' is either 'poly' for a polynomial fit of given order, or 'spline' for a Cubic Spline fit
    
    'xrange' sets the range of H+_exch. in which the interpolation is
    performed (given as fraction of 1)
    
    'show' refers to the plot
    
    Returns the result (H+_exch. as fraction of 1) and the array of the fit parameters
    
    INPUT
    val:        float
    method:     str
    x_range:    list
    show:       bool
    fit_order:  int
    mode:       str
    
    RETURN
    res:        float
    c:          array
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from .lookup import lookup_cal
    
    cal, methods = lookup_cal()
    
    if method not in methods:
        raise ValueError(f'Select method from {methods}')

    # limit interpolation to H+_exch. range
    cal = cal[(cal['H+_exch'] >= x_range[0]) & (cal['H+_exch'] <= x_range[1])]
    
    x_min = min(cal['H+_exch'])
    x_max = max(cal['H+_exch'])
    linspace = np.linspace(x_min, x_max, num=1000)
    
    x = cal['H+_exch'] # assume no x error for interpolation
    
    y = cal[method]
    y_error = cal[f'{method}_error']
    weights = 1 / y_error
        
    # perform fit
    if mode == 'poly':
        import numpy.polynomial.polynomial as P
        fit, stats = P.polyfit(x, y, deg=fit_order, full=True, w=weights)
        # checkout result for input value
        res = (P.Polynomial(fit) - val).roots()
        res = [r for r in res if r.imag == 0] # discard complex roots
        res = [r for r in res if (r > x_range[0]) & (r < x_range[1])]
        if len(res) == 1:
            res = res[0].real
        else:
            raise ValueError('Found no or more than one root')
        y_fit = P.polyval(linspace, fit)
    
    elif mode == 'spline':
        from scipy.interpolate import CubicSpline
        fit = CubicSpline(x, y)
        res = fit.solve(val) # only finds the real roots
        res = [r for r in res if (r > x_range[0]) & (r < x_range[1])]
        if len(res) == 1:
            res = res[0].real
        else:
            raise ValueError('Found no or more than one root')
        y_fit = fit(linspace)
        
    # plotting
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=y_error, color='k', fmt='s', markersize=2)
    ax.set_xlabel('$H^+_{exch.}\, [\%]$')
    if 'norm' in method:
        ax.set_ylabel(f'${method} \, [ ]$')
    elif 'HFR' in method:
        ax.set_ylabel(f'${method} \, [m\Omega cm^2]$')
    elif 'Rho' in method:
        ax.set_ylabel(f'${method} \, [\Omega cm]$')
    
    ax.plot(linspace, y_fit, color='grey')
    ax.scatter(res, val, marker='x', color='r')
    ax.set_title(f'$H^+_x = {round(res*100, 1)} \, \%$')
    
    if show:
        plt.show()
        
    return res, fit