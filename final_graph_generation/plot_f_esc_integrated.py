import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy as np
import csv
from matplotlib.ticker import MaxNLocator

# True to plot M_UV,max varying
vary_max = True

# False to plot M_UV,min varying
vary_min = False

folder = "final_graph_generation/emissivities/"
constraints_folder = 'other_paper_graphs/'

if vary_max:
    mag_range = (-10, -17)
    abs_mag_list = list(range(abs(mag_range[0]), abs(mag_range[1]) + 1))
    N_ion_esc_files_to_load = [f"thesan_N_ion_magmin_33_magmax_{i}.csv" for i in abs_mag_list]
    N_ion_prod_files_to_load = [f"thesan_N_ion_prod_magmin_33_magmax_{i}.csv" for i in abs_mag_list]
elif vary_min:
    mag_range = (-35, -25)
    abs_mag_list = list(range(abs(mag_range[1]), abs(mag_range[0]) + 1))[::-1]
    N_ion_esc_files_to_load = [f"thesan_N_ion_magmin_{i}_magmax_13.csv" for i in abs_mag_list]
    N_ion_prod_files_to_load = [f"thesan_N_ion_prod_magmin_{i}_magmax_13.csv" for i in abs_mag_list]
else:
    N_ion_esc_files_to_load = ["thesan_N_ion_magmin_33_magmax_13.csv"]
    N_ion_prod_files_to_load = ["thesan_N_ion_prod_magmin_33_magmax_13.csv"]

z_range = (1.5, 13.5)
z_space = np.linspace(z_range[0], z_range[1], 1000)

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(figsize=(10, 8))

if not vary_min and not vary_max:
    N_ion_esc_data = np.loadtxt(folder + N_ion_esc_files_to_load[0])
    redshift = N_ion_esc_data[:,0]
    log_N_ion_esc = N_ion_esc_data[:,1]
    log_N_ion_esc_err_low = N_ion_esc_data[:,2]
    log_N_ion_esc_err_high = N_ion_esc_data[:,3]

    N_ion_prod_data = np.loadtxt(folder + N_ion_prod_files_to_load[0])
    log_N_ion_prod = N_ion_prod_data[:,1]
    log_N_ion_prod_err_low = N_ion_prod_data[:,2]
    log_N_ion_prod_err_high = N_ion_prod_data[:,3]

    log_f_esc = log_N_ion_esc - log_N_ion_prod
    log_f_esc_err_low = np.sqrt(log_N_ion_esc_err_low**2 + log_N_ion_prod_err_low**2)
    log_f_esc_err_high = np.sqrt(log_N_ion_esc_err_high**2 + log_N_ion_prod_err_high**2)

    ax.errorbar(redshift, log_f_esc, yerr=(log_f_esc_err_low, log_f_esc_err_high),
                fmt='none', c='darkviolet', elinewidth=2, capsize=5, alpha=1, zorder=5)
    ax.plot(redshift, log_f_esc, linestyle='--', c='darkviolet', linewidth=2.5, alpha=1, zorder=5)
    ax.scatter(redshift, log_f_esc, s=150, c='darkviolet', edgecolors='black', zorder=6,
            label=('$\mathrm{log}_{10}(f_\mathrm{esc})$'))

else:
    for index, mag in enumerate(abs_mag_list):
        N_ion_esc_data = np.loadtxt(folder + N_ion_esc_files_to_load[index])
        redshift = N_ion_esc_data[:,0]
        log_N_ion_esc = N_ion_esc_data[:,1]
        N_ion_prod_data = np.loadtxt(folder + N_ion_prod_files_to_load[index])
        log_N_ion_prod = N_ion_prod_data[:,1]
        log_f_esc = log_N_ion_esc - log_N_ion_prod
        ax.plot(redshift, log_f_esc, linestyle='--', linewidth=2.5, alpha=1, zorder=5,
                label=f"$M_\mathrm{{UV}} < -{mag}$")
        ax.scatter(redshift, log_f_esc, s=150, edgecolors='black', alpha=1, zorder=6)

        legend = ax.legend(fontsize=20, loc='lower left', bbox_to_anchor=(0.025, 0.025), frameon=False)


ax.set_xlabel("$z$")
ax.set_ylabel("$\mathrm{log}_{10}(f_\mathrm{esc})$")
ax.set_xlim(z_range)
ax.set_ylim((-4, 0))
ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.grid(False)

plt.tight_layout()
mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.pdf", bbox_inches='tight', dpi=500)
plt.show()