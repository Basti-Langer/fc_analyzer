def get_active_area():
    # tbd read active area from customize preset
    return 5 # in cm2

def psat_H2O_calc(T):
    '''
    Returns the saturation vapor pressure of water (in kPa) at a given temperature thanks to the Antoine equation
    
    '''
    #Antoine parameters 1-100°C
    A = 8.07131
    B = 1730.63
    C = 233.426     
    psat_H2O = 10**(A-B/(C+T))
    psat_H2O = psat_H2O*133.322*1e-3 #kPa
    return(psat_H2O)

def rh_calc(T_cell, T_dew):
    rh = psat_H2O_calc(T_dew)/psat_H2O_calc(T_cell)
    return rh

def p_H2O_calc(rh,T):
    '''
    Returns the partial pressure of water (in kPa) at a given RH and temperature

    '''
    psat_H2O = psat_H2O_calc(T) #kPa
    p_H2O = rh*psat_H2O #kPa
    return p_H2O




# def p_dep_analysis(p_cell, R_tot, slope, intercept):
#   params = {a:b, c:d}


