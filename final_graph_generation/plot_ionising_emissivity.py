import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
from scipy.optimize import curve_fit
from scipy.integrate import quad, fixed_quad
import numpy as np
from astropy.io import fits
from scipy import odr
import csv
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import fftconvolve
import pickle

# True to include observational constraints in N_ion plot
include_constraints = True

# True to include Charlotte Simmonds' N_ion calculations in plot
include_charlotte = True

# True to include the original Thesan N_ion calculations in plot
include_thesan = True

# True to plot M_UV,max varying
vary_max = False

# False to plot M_UV,min varying
vary_min = False

folder = "final_graph_generation/emissivities/"
if vary_max:
    mag_range = (-10, -17)
    abs_mag_list = list(range(abs(mag_range[0]), abs(mag_range[1]) + 1))
    files_to_load = [f"thesan_N_ion_magmin_33_magmax_{i}.csv" for i in abs_mag_list]
if vary_min:
    mag_range = (-35, -25)
    abs_mag_list = list(range(abs(mag_range[1]), abs(mag_range[0]) + 1))[::-1]
    files_to_load = [f"thesan_N_ion_magmin_{i}_magmax_13.csv" for i in abs_mag_list]

z_range = (1.5, 10.5)
z_space = np.linspace(z_range[0], z_range[1], 1000)

def critical(z, C):
    # omega_b = 0.0486
    # h_50 = 100 * 0.6774 / 50
    # C_30 = C / 30
    # return 10**(51.2) * C_30 * ((1 + z) / 6)**3 * ((omega_b * h_50**2) / 0.08)**2
    Y = 0.24668
    X = 1 - Y
    chi = Y / (4 * X)
    alpha_B = 2.5775e-13 # cm^3 s^-1 at T=1e4 K
    n_H_0 = 1.8786e-7 # cm^-3
    t_rec = 1 / ((1 + chi) * alpha_B * n_H_0 * (1 + z)**3 * C)
    return n_H_0 / t_rec * (3.086e24)**3

def log_errors(errors, results):
    return errors / (results * np.log(10))

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(figsize=(20, 8))


# plot critical N_esc curves
for C in [1, 3, 10]:
    ax.plot(z_space, np.log10(critical(z_space, C)), c='grey', zorder=1)
    text_z = 2.5
    ax.text(text_z, np.log10(critical(text_z, C)) - 0.10, f'$C = {C}$', rotation=33, color='grey')
ax.fill_between(z_space, np.log10(critical(z_space, 1)), np.log10(critical(z_space, 10)),
                color='grey', alpha=0.2, zorder=1, label='Madau+1999')

if include_constraints:
    constraint_colours = {
    "becker":  "#FFA54C",
    "kuhlen": "#FF7A30",
    "davies": "#FF4C1A",
    "gaikwad": "#E22A10",
    "rinaldi": "#B21807",
    }

    # plot observational constraints on N_ion from Kuhlen (2012)
    # constraints_folder = 'other_paper_graphs/'
    # kuhlen_2012 = np.zeros((12, 4))
    # kuhlen_2012_files = ['kuhlen_2012_Nion_values.csv',
    #                       'kuhlen_2012_Nion_errors_low.csv',
    #                       'kuhlen_2012_Nion_errors_high.csv']
    # for i in range(len(kuhlen_2012_files)):
    #     with open(constraints_folder + kuhlen_2012_files[i], newline='', encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    #         rows = [row for row in reader]
    #         if i == 0:
    #             kuhlen_2012[:,0] = np.array([row[0] for row in rows]).astype('float64')
    #             kuhlen_2012[:,1] = np.array([row[1] for row in rows]).astype('float64') + 51
    #         else:
    #             kuhlen_2012[:,i+1] = np.array([row[1] for row in rows]).astype('float64') + 51
    # ax.errorbar(kuhlen_2012[:,0], kuhlen_2012[:,1],
    #             yerr=(kuhlen_2012[:,1] - kuhlen_2012[:,2], kuhlen_2012[:,3] - kuhlen_2012[:,1]),
    #             fmt='none', c='black', elinewidth=2, capsize=5, zorder=3')
    # ax.scatter(kuhlen_2012[:,0], kuhlen_2012[:,1], s=50, marker='^', c='black', edgecolors='black',
    #            zorder=4, label='kuhlen et al. (2012)')
    # ax.fill_between(kuhlen_2012[:,0], kuhlen_2012[:,2], kuhlen_2012[:,3],
    #                 color=constraint_colours["kuhlen"], alpha=0.5, zorder=2, label='Kuhlen+2012')

    # plot observational constraints on N_ion from Becker (2013)
    constraints_folder = 'other_paper_graphs/'
    becker_2013 = np.zeros((4, 4))
    becker_2013_files = ['becker_2013_Nion_values.csv',
                            'becker_2013_Nion_errors_low.csv',
                            'becker_2013_Nion_errors_high.csv']
    for i in range(len(becker_2013_files)):
        with open(constraints_folder + becker_2013_files[i], newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            rows = [row for row in reader]
            if i == 0:
                becker_2013[:,0] = np.array([row[0] for row in rows]).astype('float64')
                becker_2013[:,1] = np.array([row[1] for row in rows]).astype('float64') + 51
            else:
                becker_2013[:,i+1] = np.array([row[1] for row in rows]).astype('float64') + 51
    # ax.errorbar(becker_2013[:,0], becker_2013[:,1],
    #             yerr=(becker_2013[:,1] - becker_2013[:,2], becker_2013[:,3] - becker_2013[:,1]),
    #             fmt='none', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(becker_2013[:,0], becker_2013[:,1], s=50, marker='o', c='black', edgecolors='black',
    #            zorder=4, label='Becker et al. (2013)')
    ax.fill_between(becker_2013[:,0], becker_2013[:,2], becker_2013[:,3],
                    color=constraint_colours["becker"], alpha=0.5, zorder=2, label='Becker \& Bolton+2013')

    # plot observational constraints on N_ion from Gaikwad (2023)
    constraints_folder = 'other_paper_graphs/'
    gaikwad_2023 = np.zeros((12, 4))
    gaikwad_2023_files = ['gaikwad_2023_Nion_values.csv',
                            'gaikwad_2023_Nion_errors_low.csv',
                            'gaikwad_2023_Nion_errors_high.csv']
    for i in range(len(gaikwad_2023_files)):
        with open(constraints_folder + gaikwad_2023_files[i], newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            rows = [row for row in reader]
            if i == 0:
                gaikwad_2023[:,0] = np.array([row[0] for row in rows]).astype('float64')
                gaikwad_2023[:,1] = np.array([row[1] for row in rows]).astype('float64')
            else:
                gaikwad_2023[:,i+1] = np.array([row[1] for row in rows]).astype('float64')
    # ax.errorbar(gaikwad_2023[:,0], gaikwad_2023[:,1],
    #             yerr=(gaikwad_2023[:,1] - gaikwad_2023[:,2], gaikwad_2023[:,3] - gaikwad_2023[:,1]),
    #             fmt='o', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(gaikwad_2023[:,0], gaikwad_2023[:,1], s=50, marker='*', c='black', edgecolors='black',
    #            zorder=4, label='Gaikwad et al. (2023)')
    ax.fill_between(gaikwad_2023[:,0], gaikwad_2023[:,2], gaikwad_2023[:,3],
                    color=constraint_colours["gaikwad"], alpha=0.5, zorder=2, label='Gaikwad+2023')

    # plot observational constraints on N_ion from Davies (2024)
    # constraints_folder = 'other_paper_graphs/'
    # davies_2024 = np.zeros((5, 4))
    # davies_2024_files = ['davies_2024_Nion_values.csv',
    #                      'davies_2024_Nion_errors_low.csv',
    #                      'davies_2024_Nion_errors_high.csv']
    # for i in range(len(davies_2024_files)):
    #     with open(constraints_folder + davies_2024_files[i], newline='', encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    #         rows = [row for row in reader]
    #         if i == 0:
    #             davies_2024[:,0] = np.array([row[0] for row in rows]).astype('float64')
    #             davies_2024[:,1] = np.log10(np.array([row[1] for row in rows]).astype('float64'))
    #         else:
    #             davies_2024[:,i+1] = np.log10(np.array([row[1] for row in rows]).astype('float64'))
    # ax.errorbar(davies_2024[:,0], davies_2024[:,1],
    #             yerr=(davies_2024[:,1] - davies_2024[:,2], davies_2024[:,3] - davies_2024[:,1]),
    #             fmt='o', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(davies_2024[:,0], davies_2024[:,1], s=50, marker='*', c='black', edgecolors='black',
    #            zorder=4, label='Davies et al. (2024)')
    # ax.fill_between(davies_2024[:,0], davies_2024[:,2], davies_2024[:,3],
    #                 color=constraint_colours["davies"], alpha=0.5, zorder=2, label='Davies+2024')

    # plot obsrevational constraints on N_ion from Rinaldi (2024)
    rinaldi_2024_z = [7, 8]
    rinaldi_2024_N_ion = np.array([50.53, 50.53])
    rinaldi_2024_N_ion_err = np.array([0.45, 0.45])
    ax.fill_between(rinaldi_2024_z,
                    rinaldi_2024_N_ion - rinaldi_2024_N_ion_err, rinaldi_2024_N_ion + rinaldi_2024_N_ion_err,
                    color=constraint_colours['rinaldi'], alpha=0.5, zorder=2, label='Rinaldi+2024')

if include_charlotte:
    # plot Charlotte's N_ion integrations for her f_esc = 10% and f_esc Chisholm (2022) prescriptions
    charlotte_z = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    N_ion_f_esc_10 = [51.23478634, 50.85474357, 50.60201125, 50.57420019, 50.34034352, 49.99205213]
    N_ion_f_esc_chisholm = [50.76232333, 50.52949668, 50.52709877, 50.61663425, 50.36362092, 49.44427493]
    ax.errorbar(charlotte_z, N_ion_f_esc_10, yerr=0.3,
                fmt='none', c='coral', elinewidth=2, alpha=0.8, capsize=0, zorder=2)
    ax.plot(charlotte_z, N_ion_f_esc_10, linestyle=':', c='coral', linewidth=2.5, alpha=0.8, zorder=2)
    ax.scatter(charlotte_z, N_ion_f_esc_10, s=150, marker='^', c='coral', edgecolors='black', zorder=3,
            label='Simmonds+2024, $f_\mathrm{esc}$ = 10\%')
    ax.errorbar(charlotte_z, N_ion_f_esc_chisholm, yerr=0.3,
                fmt='none', c='yellowgreen', elinewidth=2, alpha=0.8, capsize=0, zorder=2)
    ax.plot(charlotte_z, N_ion_f_esc_chisholm, linestyle=':', c='yellowgreen', linewidth=2.5, alpha=0.8, zorder=2)
    ax.scatter(charlotte_z, N_ion_f_esc_chisholm, s=150, marker='v', c='yellowgreen', edgecolors='black', zorder=3,
            label='Simmonds+2024, $f_\mathrm{esc}$ = Chisholm+2022')

if include_thesan:
    constraints_folder = 'other_paper_graphs/'
    thesan_files = ['thesan1_Nion_values.csv',
                    'thesan2_Nion_values.csv']
    with open(constraints_folder + thesan_files[0], newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        thesan1 = np.transpose(np.array([row for row in reader])).astype('float64')
        thesan1[1] = np.log10(thesan1[1]) + 51
    ax.plot(thesan1[0], thesan1[1], linestyle='-', c='red', linewidth=2.5, alpha=0.8, zorder=3, label='Garaldi+22, Thesan-1')
    with open(constraints_folder + thesan_files[1], newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        thesan2 = np.transpose(np.array([row for row in reader])).astype('float64')
        thesan2[1] = np.log10(thesan2[1]) + 51
    ax.plot(thesan2[0], thesan2[1], linestyle='-', c='blue', linewidth=2.5, alpha=0.8, zorder=3, label='Garaldi+22, Thesan-2')

if vary_max or vary_min:
    for index, mag in enumerate(abs_mag_list):
        data = np.loadtxt(folder + files_to_load[index])
        redshift = data[:,0]
        log_N_ion = data[:,1]
        log_N_ion_err_low = data[:,2]
        log_N_ion_err_high = data[:,3]
        label = [r'$\textsc{thesan-zoom}$-based $M_\mathrm{UV, min} = $' + f'-{mag}',
                r'$\textsc{thesan-zoom}$-based $M_\mathrm{UV, max} = $' + f'-{mag}'][vary_max]
        # ax.errorbar(redshift, log_N_ion, yerr=(log_N_ion_err_low, log_N_ion_err_high),
        #     fmt='none', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
        ax.plot(redshift, log_N_ion, linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(redshift, log_N_ion, s=150, edgecolors='black', zorder=5, label=label)

else:
    # plot this work's Thesan-Zoom derived Schechter UV magnitude N_esc integrations
    thesan_data = np.loadtxt(folder + "thesan_N_ion_magmin_33_magmax_13.csv")
    thesan_redshift = thesan_data[:,0]
    thesan_log_N_ion = thesan_data[:,1]
    thesan_log_N_ion_err_low = thesan_data[:,2]
    thesan_log_N_ion_err_high = thesan_data[:,3]
    ax.errorbar(thesan_redshift, thesan_log_N_ion, yerr=(thesan_log_N_ion_err_low, thesan_log_N_ion_err_high),
                fmt='none', c='darkviolet', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
    ax.plot(thesan_redshift, thesan_log_N_ion, linestyle='--', c='darkviolet', linewidth=2.5, alpha=0.8, zorder=2)
    ax.scatter(thesan_redshift, thesan_log_N_ion, s=150, c='darkviolet', edgecolors='black', zorder=5,
            label=(r'$\textsc{thesan-zoom}$-based (This Work)'))

    # plot this work's observational derived Schechter UV magnitude N_esc integrations
    observational_data = np.loadtxt(folder + "observational_N_ion.csv")
    observational_redshift = observational_data[:,0]
    observational_log_N_ion = observational_data[:,1]
    observational_log_N_ion_err_low = observational_data[:,2]
    observational_log_N_ion_err_high = observational_data[:,3]
    ax.errorbar(observational_redshift, observational_log_N_ion, yerr=(observational_log_N_ion_err_low, observational_log_N_ion_err_high),
                fmt='none', c='darkcyan', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
    ax.plot(observational_redshift, observational_log_N_ion, linestyle='--', c='darkcyan', linewidth=2.5, alpha=0.8, zorder=2)
    ax.scatter(observational_redshift, observational_log_N_ion, s=150, c='darkcyan', edgecolors='black', zorder=5,
            label='JWST-based (This Work)')

ax.set_xlabel("$z$")
ax.set_ylabel("$\mathrm{log}_{10}(\dot{N}_\mathrm{ion} \; [\mathrm{s^{-1} \; cMpc^{-3}}])$")
ax.yaxis.set_label_coords(-0.075, 0.5)
ax.set_xlim(z_range)
ax.set_ylim((48.9, 51.6))
ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
ax.grid(False)
# ax.grid(True, alpha=0.8, linestyle='--')
# ax.set_axisbelow(True)
# for line in ax.get_xgridlines() + ax.get_ygridlines():
#     line.set_zorder(0)
legend = ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1.025, 0.5), borderaxespad=0)
frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_boxstyle('Square')
frame.set_alpha(0.8)
plt.subplots_adjust(right=0.7)

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()