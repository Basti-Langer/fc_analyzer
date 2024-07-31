from impedance.models.circuits.elements import element

@element(2, units=['H', ''])
def Lmod(x, f): # x = [l, beta]
    '''
    Modified inductor (for usage with impedance.py module)
    '''
    import numpy as np
    return x[0]*((1j*2*np.pi*np.array(f))**x[1]) # in Ohm, complex number 

@element(4, units=['Ohm', 'F/cm3', '', 'Ohm'])
def TLMRct(x, f):
    '''
    f: frequencies
    x: parameters
    x = [R_cath, Q, alpha, R_ct]
    e.g. see: DOI 10.1149/1945-7111/ac1812
    '''
    import numpy as np
    
    B = np.sqrt(x[0] * x[1] * (1j*2* np.pi * np.array(f))**x[2] + x[0]/x[3])
    return x[0] / B * (np.cosh(B) / np.sinh(B)) # in Ohm, complex number

@element(3, units=['S/cm2', 'F/cm2', ''])
def Rexgrad(x, f):
    '''
    
    based on Reshetenko, Kulikovsky 2017
    - Impedance Spectroscopy Study of the PEM Fuel Cell Cathode with Nonuniform Nafion Loading
    
    Exponentially decaying conductivity profile
    
    x = [sigma0, C_dl, gamma]
    in units of ['S/cm2', 'F/cm2', '']
    'gamma' is exp. decay parameter (beta in original paper)
    
    Set thickness t = 1 cm yields area normalized values
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import jv, yv
    
    Omega = 2*np.pi*np.array(f) * x[1] / x[0]
    term1 = 2 / x[2] * np.sqrt(-1j * Omega)  # q in paper
    term2 = term1 * np.exp(x[2]/2)
    return np.sqrt(1j / Omega) * ((yv(0, term2) * jv(1, term1)) - (yv(1, term1) * jv(0, term2))) / ((yv(0, term1) * jv(0, term2)) - (yv(0, term2) * jv(0, term1)))

    
# linear tortuosity gradient (Bharat) here

def inductor(f, x): # f: frequency, x: [l, beta]
    '''
    Modified inductor (general purpose)
    f: frequencies
    x: parameters
    '''
    import numpy as np
    return x[0]*((1j*2*np.pi*f)**x[1]) # in Ohm, complex number 

def TLM_blocking(f, x): 
    '''
    f: frequencies
    x: parameters
    e.g. see: DOI 10.1149/1945-7111/ac1812
    '''
    import numpy as np
    # x = [hfr, r_cath, alpha, beta, q, l]
    A = np.sqrt(x[1] * x[4] * (1j*2* np.pi * f)**x[2])
    return np.sqrt(x[1] / (x[4] * (1j*2* np.pi * f)**x[2])) * (np.cosh(A) / np.sinh(A)) + x[5]*(1j*2*np.pi*f)**x[3] + x[0] # in Ohm, complex number

def TLM_load(f, x):
    '''
    f: frequencies
    x: parameters
    e.g. see: DOI 10.1149/1945-7111/ac1812
    '''
    import numpy as np
    # x = [hfr, r_cath, alpha, beta, q, l, r_ct]
    B = np.sqrt(x[1] * x[4] * (1j*2* np.pi * f)**x[2]) + x[1]/x[6]
    return np.sqrt(x[1] / B) * (np.cosh(B) / np.sinh(B)) + x[5]*(1j*2*np.pi*f)**x[3] + x[0] # in Ohm, complex number