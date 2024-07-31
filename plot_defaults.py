import matplotlib.pyplot as plt

def set_default_params():
    '''
    Define standard plotting parameters
    See matplotlib webpage for details
    '''

    rcdict =    {
                    'figure.figsize':       (9, 9),
                    'font.size':            12,
                    'lines.linewidth':      2.,
                    'lines.markersize':     8,
                    
                    'scatter.marker':       's',
                    
                    'font.family':          'sans-serif',
                    #'font.sans-serif':      'Computer Modern Sans Serif',
                    # 'mathtext.fontset':     'cm',
                    # 'mathtext.fontset':     'stixsans',
                    'mathtext.fontset':     'dejavusans',

                    'xtick.direction':      'in',
                    'xtick.major.size':     8.,
                    'xtick.minor.size':     3.,
                    'xtick.minor.visible':  True,
                    'xtick.top':            True,

                    'ytick.direction':      'in',
                    'ytick.major.size':     8.,
                    'ytick.minor.size':     3.,
                    'ytick.minor.visible':  True,
                    'ytick.right':          True,
                    
                    # 'grid.linestyle':       '-',
                    # 'grid.alpha':           0.75,
                    
                    'legend.framealpha':    1,
                    'legend.markerscale':   1.5,

                    'axes.linewidth':       1.5,
                    # 'axes.grid':            True,
                    'axes.labelsize':       14,
                    
                    'savefig.format':       'png',

                }

    plt.rcParams.update(rcdict)
    return None

