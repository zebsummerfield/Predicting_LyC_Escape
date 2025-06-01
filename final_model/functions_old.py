import numpy as np
from matplotlib import rcParams

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

def configure_plots():
    """
    Setting global Matplotlib settings for plotting lineplots
    just run configure_plots()

    """
    # line settings
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 3
    rcParams['errorbar.capsize'] = 3

    # tick settings
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['xtick.major.size'] = 7
    rcParams['xtick.minor.size'] = 4
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.major.size'] = 7
    rcParams['ytick.minor.size'] = 4
    rcParams['ytick.direction'] = 'in'

    # text settings
    rcParams['mathtext.rm'] = 'serif'
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['text.usetex'] = False
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 16
    rcParams['axes.ymargin'] = 0.5

    # legend
    rcParams['legend.fontsize'] = 12
    rcParams['legend.frameon'] = True

    # grid in plots
    rcParams['grid.linestyle'] = ':'

    # figure settings
    rcParams['figure.figsize'] = 5, 4