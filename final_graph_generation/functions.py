import numpy as np
import h5py
from matplotlib import rcParams
import random
import scipy.optimize as so
import scipy.stats as ss
from sklearn.utils import resample

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

tableau20 = {'sfr10': (31/255, 119/255, 180/255),   # Blue 
             'sfr100': (174/255, 199/255, 232/255),  # Light Blue
             'offset10': (255/255, 127/255, 14/255),   # Orange
             'ssfr10/ssfr100': (255/255, 187/255, 120/255),  # Light Orange
             'sfr10_density': (44/255, 160/255, 44/255),    # Green
             'star_mass': (152/255, 223/255, 138/255),  # Light Green
             'gas_mass/star_mass': (214/255, 39/255, 40/255),    # Red
             'star_mass/vir_mass': (255/255, 152/255, 150/255),  # Light Red
             'gas_met': (148/255, 103/255, 189/255),  # Purple
             'uv_mag': (197/255, 176/255, 213/255),  # Light Purple
             'ha_mag':(140/255, 86/255, 75/255),    # Brown
             'uv_lum/ha_lum': (196/255, 156/255, 148/255),  # Light Brown
            'uv_size': (227/255, 119/255, 194/255),  # Pink 
            'ha_size': (247/255, 182/255, 210/255),  # Light Pink
            'sfr_size': (127/255, 127/255, 127/255),  # Gray
            'sfr_size/star_size': (199/255, 199/255, 199/255),  # Light Gray
            '1 + redshift': (188/255, 189/255, 34/255),   # Olive
            'random_variable': (219/255, 219/255, 141/255),  # Light Olive
            # (23/255, 190/255, 207/255),   # Teal
            # (158/255, 218/255, 229/255)   # Light Teal
            }

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

def prepare_data(file, f_or_n=0, obvs=False, eps=True, add_vars=[]):
    """
    Loads the data from the training catalogue and prepares it for model training

    file: str
        The path to the hdf5 file containing the training data
    f_or_n: int
        0 for f_esc, 1 for n_esc
    obvs: bool
        True if model is generated to predict for an observational catalogue
    eps: bool
        True if the variables should be adjusted to avoid log(0) errors
    ad_vars: list
        A list of additional variables to add to the training data
    """

    # loads both the catalogue of galaxies and their variables
    with h5py.File(file, 'r') as hdf:

        f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
        n_esc = np.array(hdf['Ndot_LyC_vir_full'])

        redshift = np.array(hdf['redshift_full'])
        star_mass = np.array(hdf['stellar_mass_full'])
        sfr10 = np.array(hdf['sfr_full_10'])
        sfr50 = np.array(hdf['sfr_full_50'])
        sfr100 = np.array(hdf['sfr_full_100'])
        ssfr10 = ssfr_func(sfr10, star_mass)
        ssfr50 = ssfr_func(sfr50, star_mass)
        ssfr100 = ssfr_func(sfr100, star_mass)
        vir_mass = np.array(hdf['M_vir_full'])
        gas_mass = np.array(hdf['gas_mass_full'])
        ha_lum = np.array(hdf['ha_lum_int_full']) * 5
        uv_lum = np.array(hdf['uv_lum_int_full'])
        # gas_met = np.array(hdf['gas_met_full'])
        star_met = np.array(hdf['star_met_full'])
        star_size = np.array(hdf['stellar_size_full'])
        sfr_size = np.array(hdf['sfr_size_full'])
        ha_size = np.array(hdf['ha_size_int_full'])
        uv_size = np.array(hdf['uv_size_int_full'])
        uv_projection = np.array(hdf['uv_size_int_2d_full'])
        ha_projection = np.array(hdf['ha_size_int_2d_full'])
        sfr10 = np.array(hdf['sfr_full_10'])
        sfr10_density = sfr10 / (np.pi * sfr_size**2)
        # fixing gas mass units
        gas_mass = gas_mass / (0.76 / 1.6735575e-24)
        gas_mass = gas_mass / 1.989e33
        
        add_vars_list = []
        for str in add_vars:
            add_vars_list.append(np.array(hdf[str]))

        random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

        f_esc_vars = np.array([ssfr10, ssfr100, star_mass, gas_mass, vir_mass, star_met,
                            uv_lum, ha_lum, uv_size, ha_size,
                            sfr_size, star_size, sfr10_density, (1+redshift), random_variable,])
        f_esc_keys = np.array(['offset10', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass', 'star_mass/vir_mass',
                            'gas_met', 'uv_mag', 'uv_lum/ha_lum', 'uv_size', 'ha_size',
                            'sfr_size', 'sfr_size/star_size', 'sfr10_density', '1 + redshift', 'random_variable'])
        n_esc_vars = np.array([sfr10, sfr100, star_mass, gas_mass, vir_mass, star_met, 
                            uv_lum, ha_lum, (1+redshift), random_variable])
        n_esc_keys = np.array(['sfr10', 'sfr100', 'star_mass','gas_mass/star_mass', 'star_mass/vir_mass', 'gas_met', 
                            'uv_mag', 'ha_mag', '1 + redshift', 'random_variable'])
        
        # adds a small epsilon to the variables to avoid log(0) errors
        if eps:
            eps_frac = 0.01
            f_epsilons = np.array([min([v for v in var if v !=0])*eps_frac for var in f_esc_vars])
            eps_array = np.zeros(f_esc_vars.shape)
            for i in range(len(eps_array)):
                eps_array[i].fill(f_epsilons[i])
            f_esc_vars = f_esc_vars + eps_array
            n_epsilons = np.array([min([v for v in var if v !=0])*eps_frac for var in n_esc_vars])
            eps_array = np.zeros(n_esc_vars.shape)
            for i in range(len(eps_array)):
                eps_array[i].fill(n_epsilons[i])
            n_esc_vars = n_esc_vars + eps_array

        # post adding epsilon, the variables are changed to match the desired form given by their key
        f_esc_vars[1] = f_esc_vars[0] / f_esc_vars[1]
        f_esc_vars[3] = f_esc_vars[3] / f_esc_vars[2]
        f_esc_vars[4] = f_esc_vars[2] / f_esc_vars[4]
        f_esc_vars[7] = f_esc_vars[6] / f_esc_vars[7]
        f_esc_vars[11] = f_esc_vars[10] / f_esc_vars[11]
        n_esc_vars[3] = n_esc_vars[3] / n_esc_vars[2]
        n_esc_vars[4] = n_esc_vars[2] / n_esc_vars[4]

        # replaces the sizes with 2d projections to match the observational catalogue
        if obvs:
            f_esc_vars[8] = uv_projection
            f_esc_vars[9] = ha_projection
        
        log_f_esc_vars = np.log10(f_esc_vars).astype('float32')
        log_n_esc_vars = np.log10(n_esc_vars).astype('float32')

        # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
        log_osfms10 =  log_f_esc_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
        log_f_esc_vars[0] = log_osfms10.astype('float32')
        # replaces the luminosities with magnitudes
        lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
        uv_mag = -2.5 * np.log10((n_esc_vars[6]) / lum_to_tenpc) - 48.6
        ha_mag = -2.5 * np.log10((n_esc_vars[7]) / lum_to_tenpc) - 48.6
        log_f_esc_vars[6] = uv_mag.astype('float32')
        log_n_esc_vars[6] = uv_mag.astype('float32')
        log_n_esc_vars[7] = ha_mag.astype('float32')

        vars = [f_esc_vars, n_esc_vars][f_or_n]
        log_vars = [log_f_esc_vars, log_n_esc_vars][f_or_n]
        keys = [f_esc_keys, n_esc_keys][f_or_n]

        # selects only the variables that are present in the observational catalogue
        f_esc_observational_vars = [0, 1, 2, 6, 7, 8, 9, 13, 14]
        n_esc_observational_vars = [0, 1, 2, 6, 7, 8, 9]
        selected_vars = [f_esc_observational_vars, n_esc_observational_vars][f_or_n]
        if obvs:
            keys = keys[selected_vars]
            vars = vars[selected_vars]
            log_vars = log_vars[selected_vars]
        if add_vars:
            keys = np.concatenate((keys, add_vars))
            vars = np.concatenate((vars, add_vars_list))
            log_vars = np.concatenate((log_vars, np.log10(add_vars_list).astype('float32')))
        print(keys)
        
        # removes any rows that have zero ,unity or infinity for the vars and f_esc
        # ssfr50 is included to remove galaxies that have had no recent star formation
        f_esc[np.isnan(f_esc)] = 0
        for i in range(len(np.concatenate((log_vars, [f_esc], [ssfr50])))):
            nan_indices = [index for index, val in enumerate(list(np.concatenate((log_vars, [f_esc], [ssfr50]))[i]))
                        if (val == 0 or val == 1 or val == np.inf or val== -np.inf)][::-1]
            print(f"rows deleted: {len(nan_indices)}")
            f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
            log_vars = np.delete(log_vars, nan_indices, axis=1)
            ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, nan_indices),
                                       np.delete(ssfr50, nan_indices),
                                       np.delete(ssfr100, nan_indices))
            
        print(f'rows remaining: {len(f_esc)}')
        log_f_esc = np.log10(f_esc).astype('float32')
        log_n_esc = np.log10(n_esc).astype('float32')
    
    return (keys, log_vars, log_f_esc, log_n_esc)


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def make_contours(x,y, nbins=50, nlevels=None) :

    density_points, x_edges, y_edges = np.histogram2d(x, y, bins=(nbins,nbins), density=True)
    x_bin_sizes = (x_edges[1:] - x_edges[:-1]).reshape((1,nbins))
    y_bin_sizes = (y_edges[1:] - y_edges[:-1]).reshape((nbins,1))
    X, Y = 0.5*(x_edges[1:]+x_edges[:-1]), 0.5*(y_edges[1:]+y_edges[:-1])

    pdf = (density_points*(x_bin_sizes*y_bin_sizes))
    zero_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.01))
    # brentq finds root of function (find_...) between x=0 and 1, with arguments
    # pdf and 0.01 as confidence level here.
    
    zero_sigma1 = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.34))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.84))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    four_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.975))
    five_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    #you can select here which levels you are interested in
    levels = [zero_sigma1, one_sigma, two_sigma,three_sigma, four_sigma][::-1]
    if nlevels is not None :
     levels = levels[nlevels:]
    
    return [X,Y, pdf,levels]

def pca_n(vari):
    dim=len(vari)
    if dim==3:

        # First we need the Spearmann rank correlation coefficients 
        
        
        r_AB=np.zeros((len(vari), len(vari)))
        indices=np.arange(0, len(vari), 1)
        
        for i in indices:
            for j in indices[(indices!=i)]:
                r_ij=ss.spearmanr(vari[i], vari[j])
                r_AB[i,j]=r_ij[0]
        
        # now to make the PCA matrix
                
        r_AB_C=np.zeros((len(vari), len(vari)))
        
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    r_ij_k=(r_AB[i,j]-r_AB[i,k]*r_AB[j,k])/(
                        np.sqrt(1-r_AB[i,k]**2)*np.sqrt(1-r_AB[j,k]**2))
                    r_AB_C[i,j]=r_ij_k 
                    
        tantheta=r_AB_C[0,2]/r_AB_C[1,2]
        return r_AB_C, tantheta
    
    elif dim==4:
        
        # try with 4 variables 
                
        r_AB=np.zeros((len(vari), len(vari)))
        
        indices=np.arange(0, len(vari), 1)
        
        for i in indices:
            for j in indices[(indices!=i)]:
                r_ij=ss.spearmanr(vari[i], vari[j])
                r_AB[i,j]=r_ij[0]
               
        
        r_AB_C=np.zeros((len(vari), len(vari), len(vari)))
        
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    r_ij_k=(r_AB[i,j]-r_AB[i,k]*r_AB[j,k])/(
                        np.sqrt(1-r_AB[i,k]**2)*np.sqrt(1-r_AB[j,k]**2))
                    r_AB_C[i,j,k]=r_ij_k
            
        r_AB_CD=np.zeros((len(vari), len(vari)))
             
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    l=indices[(indices!=i) & (indices!=j) & (indices!=k)]
        
                    r_ij_kl=(r_AB_C[i,j,k]-r_AB_C[i,l,k]*r_AB_C[j,l,k])/(
                                np.sqrt(1-r_AB_C[i,l,k]**2)*np.sqrt(1-r_AB_C[j,l,k]**2))
                    
                    r_AB_CD[i,j]=r_ij_kl
            tantheta=0
        return r_AB_CD, tantheta 

def pcc_err(vari_pcc, theta=False):

    n_s=100 
    # repeat n_s times
    
    s_s=len(vari_pcc[0]) 
    # draw s_s elements with replacement from sample each time i.e. its put back so can be picked again
    
    random.seed(10)
    dat=vari_pcc.T
    pcc_sam=[[]]
    theta_sam=[]
    
    for i in range(n_s):
        dat_sam=resample(dat, n_samples=s_s, random_state=i)  
        r_AB_C, tantheta=pca_n(dat_sam.T)
        pcc_sam.append(r_AB_C.ravel())
        theta_sam.append(np.arctan(tantheta))
    
    del pcc_sam[0]
    
    pcc_std_l=np.std(pcc_sam, axis=0)
    
    dim=len(vari_pcc)
    pcc_std=pcc_std_l.reshape(dim,dim)
    r_AB_C, tantheta=pca_n(vari_pcc)
    
    if theta:
        theta_err=np.std(theta_sam)
        return np.arctan(tantheta), theta_err
    # returns theta of normal PCC and then error from bootstrapping method!
    
    else:
        print(r_AB_C)
        print(pcc_std)
        return r_AB_C, pcc_std