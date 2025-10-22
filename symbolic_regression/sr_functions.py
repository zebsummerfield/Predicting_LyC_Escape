import h5py
import numpy as np
import sympy as sp

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

def prepare_data_sr(file, f_or_n=0, basic=False, obvs=False, obvs_cat='charlotte', eps=True, add_vars=[]):
    """
    Loads the data from the training catalogue and prepares it for model training

    file: str
        The path to the hdf5 file containing the training data
    f_or_n: int
        0 for f_esc, 1 for n_esc
    obvs: bool
        True if model is generated to predict for an observational catalogue
    obvs_cat: str
        The name of the observational catalogue being used ('charlotte' or 'lola')
    eps: bool
        True if the variables should be adjusted to avoid log(0) errors
    add_vars: list
        A list of additional variables to add to the training data
    """

    # loads both the catalogue of galaxies and their variables
    with h5py.File(file, 'r') as hdf:

        f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
        n_esc = np.array(hdf['Ndot_LyC_vir_full'])

        resolution = np.array([zoom.decode('utf-8') for zoom in hdf['zoomlevel_full']])
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
        uv_size = uv_projection
        ha_size = ha_projection
        sfr10 = np.array(hdf['sfr_full_10'])
        sfr10_density = sfr10 / (np.pi * sfr_size**2)
        # fixing gas mass units
        gas_mass = gas_mass / (0.76 / 1.6735575e-24)
        gas_mass = gas_mass / 1.989e33
        no_gas_indices = [index for index, val in enumerate(gas_mass) if val <= 0]
        
        add_vars_list = []
        for str in add_vars:
            add_vars_list.append(np.array(hdf[str]))

        random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

        f_esc_vars = np.array([ssfr10, ssfr100, ssfr100, star_mass, gas_mass, vir_mass, star_met, uv_lum, ha_lum,
                               uv_size, ha_size, sfr_size, star_size, sfr10_density, (1+redshift), random_variable])
        f_esc_keys = np.array(['offset10', 'ssfr100', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass', 'star_mass/vir_mass',
                            'star_met', 'uv_mag', 'uv_lum/ha_lum', 'uv_size', 'ha_size',
                            'sfr_size', 'sfr_size/star_size', 'sfr10_density', '1 + redshift', 'random_variable'])
        n_esc_vars = np.array([sfr10, ssfr100, sfr100, star_mass, gas_mass, vir_mass, star_met, 
                            uv_lum, ha_lum, (1+redshift), random_variable])
        n_esc_keys = np.array(['sfr10', 'sfr100', 'star_mass','gas_mass/star_mass', 'star_mass/vir_mass', 'star_met', 
                            'uv_mag', 'ha_mag', '1 + redshift', 'random_variable'])
        basic_vars = np.array([sfr10, sfr100, star_mass, gas_mass, vir_mass, star_met, uv_lum, ha_lum,
                               uv_size, ha_size, sfr_size, star_size, redshift, random_variable])
        basic_keys = np.array(['sfr10', 'sfr100', 'star_mass','gas_mass', 'vir_mass', 'star_met', 'uv_lum', 'ha_lum',
                               'uv_size', 'ha_size', 'sfr_size', 'star_size', 'redshift', 'random_variable'])
        
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
            b_epsilons = np.array([min([v for v in var if v !=0])*eps_frac for var in basic_vars])
            eps_array = np.zeros(basic_vars.shape)
            for i in range(len(eps_array)):
                eps_array[i].fill(b_epsilons[i])
            basic_vars = basic_vars + eps_array

        # post adding epsilon, the variables are changed to match the desired form given by their key
        f_esc_vars[2] = f_esc_vars[0] / f_esc_vars[1]
        f_esc_vars[4] = f_esc_vars[4] / f_esc_vars[3]
        f_esc_vars[5] = f_esc_vars[3] / f_esc_vars[5]
        f_esc_vars[8] = f_esc_vars[7] / f_esc_vars[8]
        f_esc_vars[12] = f_esc_vars[11] / f_esc_vars[12]
        n_esc_vars[3] = n_esc_vars[3] / n_esc_vars[2]
        n_esc_vars[4] = n_esc_vars[2] / n_esc_vars[4]
        
        log_f_esc_vars = np.log10(f_esc_vars).astype('float32')
        log_n_esc_vars = np.log10(n_esc_vars).astype('float32')
        log_basic_vars = np.log10(basic_vars).astype('float32')

        # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
        log_osfms10 =  log_f_esc_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
        log_f_esc_vars[0] = log_osfms10.astype('float32')
        # replaces the luminosities with magnitudes
        lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
        uv_mag = -2.5 * np.log10((n_esc_vars[6]) / lum_to_tenpc) - 48.6
        ha_mag = -2.5 * np.log10((n_esc_vars[7]) / lum_to_tenpc) - 48.6
        log_f_esc_vars[7] = uv_mag.astype('float32')
        log_n_esc_vars[6] = uv_mag.astype('float32')
        log_n_esc_vars[7] = ha_mag.astype('float32')

        # selects only the variables that are present in the observational catalogue
        if obvs_cat == 'lola':
            f_esc_observational_vars = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15]
            n_esc_observational_vars = [0, 1, 2, 6, 7, 8, 9]
        else:
            f_esc_observational_vars = [0, 1, 2, 3, 7, 14, 15]        
            n_esc_observational_vars = [0, 1, 2, 6, 8, 9]

        var_list_index = ((f_or_n), 2)[basic]
        vars = [f_esc_vars, n_esc_vars, basic_vars][var_list_index]
        log_vars = [log_f_esc_vars, log_n_esc_vars, log_basic_vars][var_list_index]
        keys = [f_esc_keys, n_esc_keys, basic_keys][var_list_index]
        selected_vars = [f_esc_observational_vars, n_esc_observational_vars, [0, 1, 2, 6, 12]][var_list_index]

        if obvs:
            keys = keys[selected_vars]
            vars = vars[selected_vars]
            log_vars = log_vars[selected_vars]
        print(keys)
        
        # removes any rows that have zero ,unity, nan or infinity for the vars and f_esc
        # ssfr50 is included to remove galaxies that have had no recent star formation
        f_esc[np.isnan(f_esc)] = 0
        bad_indices = no_gas_indices
        print(f"0 gas mass rows removed: {len(no_gas_indices)}")
        for i in range(len(np.concatenate((log_vars, [f_esc], [ssfr50])))):
            b_i = [index for index, val in enumerate(list(np.concatenate((log_vars, [f_esc], [ssfr50]))[i]))
                            if (val == 0 or val == 1 or val == np.inf or val== -np.inf or np.isnan(val))]
            print(f"feature {i+1} bad rows: {len(b_i)}")
            bad_indices += b_i
        # b_i = [index for index, zoom in enumerate(resolution) if zoom != 'z4']
        # print(f"zoom level bad rows: {len(b_i)}")
        # bad_indices += b_i
        bad_indices = list(set(bad_indices))[::-1]
        f_esc, n_esc = (np.delete(f_esc, bad_indices), np.delete(n_esc, bad_indices))
        log_vars = np.delete(log_vars, bad_indices, axis=1)
        ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, bad_indices),
                                    np.delete(ssfr50, bad_indices),
                                    np.delete(ssfr100, bad_indices))
        resolution = np.delete(resolution, bad_indices)

        print(f'rows removed: {len(bad_indices)}')
        print(f'rows remaining: {len(f_esc)}')
        log_f_esc = np.log10(f_esc).astype('float32')
        log_n_esc = np.log10(n_esc).astype('float32')
        print(f"mean f_esc: {np.mean(f_esc)}")
        print(f"mean n_esc: {np.mean(n_esc)}")
    
    return (keys, log_vars, (log_f_esc, log_n_esc)[f_or_n], resolution)

def simplify_model(expr):
    """
    Convert a SymPy expression from log10-space variables (x0, x1, etc.)
    to original physical variables (X0, X1, etc.) and simplify 10**(...) 
    into a product-of-powers form.
    """
    # Find all free symbols in the expression
    free_symbols = expr.free_symbols
    subs_dict = {}
    
    for sym in free_symbols:
        name = str(sym)
        X_sym = sp.symbols(f'X{name[1:]}')  # create corresponding original variable Xn
        subs_dict[sym] = sp.log(X_sym, 10) # substitute x_n -> log10(X_n)
    
    # Substitute log10 variables with original variables
    expr_original = expr.subs(subs_dict)
    
    # Simplify 10**(...) into product-of-powers
    expr_simplified = sp.simplify(10**expr_original)
    
    return expr_simplified
            