import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sympy import Line2D
from functions import *
import numpy as np
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.integrate import solve_ivp, cumtrapz

# Set to True to plot ionisation fraction Q, False to plot optical depth tau
Q_plot = True

# Set to True to compare Q evolutions for different sigma_M_uv values
sigma_compare = False

# Set to True to compare Q evolutions for different N_ion integration M_uv_max values
muvmax_compare = True

# Set to True to compare Q evolutions for different N_ion integration M_uv_min values
muvmin_compare = False

# Set to True to include f_esc constant splines when comparing muvmax
include_f_esc_const = True

Y = 0.24668
X = 1 - Y
chi = Y / (4 * X)
alpha_B = 2.5775e-13 # cm^3 s^-1 at T=1e4 K
n_H_0 = 1.8786e-7 # cm^-3
omega_m = 0.3089
omega_lambda = 0.6911
H_0 = 2.195e-18 # s^-1
c = 2.998e10 # cm/s
sigma_T = 6.65246e-25 # cm^2

folder = "final_graph_generation/splines/"
with open(folder + "thesan_uv_N_ion_spline_sch.pkl", "rb") as f:
    thesan_spline = pickle.load(f)
with open(folder + "thesan_uv_N_ion_spline_high_sch.pkl", "rb") as f:
    thesan_spline_high = pickle.load(f)
with open(folder + "thesan_uv_N_ion_spline_low_sch.pkl", "rb") as f:
    thesan_spline_low = pickle.load(f)
with open(folder + "observational_uv_N_ion_spline_sch.pkl", "rb") as f:
    observational_spline = pickle.load(f)
with open(folder + "observational_uv_N_ion_spline_high_sch.pkl", "rb") as f:
    observational_spline_high = pickle.load(f)
with open(folder + "observational_uv_N_ion_spline_low_sch.pkl", "rb") as f:
    observational_spline_low = pickle.load(f)

def N_ion(z,spl):
    return 10**spl(z) * (1/3.086e24)**3  # convert from Mpc^-3 to cm^-3

def n_H(z):
    return n_H_0 * (1 + z)**3

def t_rec(z):
    C = 9.25 - 7.21 * np.log10(1 + z) # Madau et al. 2024
    # C = 2.9 * ((1 + z)/6)**-1.1 # Shull et al. 2012
    return 1 / ((1 + chi) * alpha_B * n_H(z) * C)

def H(z):
    return H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def dQ_dz(z, Q, spl):
    return - (1 / (H(z) * (1 + z))) * ((N_ion(z, spl) / n_H_0) - (Q / t_rec(z)))

def helium(z):
    return np.array([(1, 2)[z_i <= 4] for z_i in z])

def tau(q_sol):
    # computes the optical depth to Thomson scattering up to redshift z
    z = np.array(q_sol.t)
    Q = np.array(q_sol.y[0])
    Q[Q > 1] = 1
    idx = np.argsort(z)
    z = z[idx]
    Q = Q[idx]
    const = c * sigma_T * n_H_0
    integrand = (1 + z)**2 * Q / H(z) * (1 + helium(z)*chi)
    # import pdb; pdb.set_trace()
    tau = const * cumtrapz(integrand, z, initial=0)
    return z, tau

# loads the N_ion splines and solves the ODE for ionisation fraction as a function of redshift
if sigma_compare:
    splines = []
    sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4][::-1]
    for sigma in sigmas:
        with open(folder + f"observational_uv_N_ion_spline_sch_{str(sigma)}.pkl", "rb") as f:
            sigma_spline = pickle.load(f)
        splines.append(sigma_spline)

elif muvmax_compare:
    splines = []
    mag_range = (-10, -17)
    abs_mag_list = list(range(abs(mag_range[0]), abs(mag_range[1]) + 1))
    for mag in abs_mag_list:
        with open(folder + f"thesan_N_ion_magmin_33_magmax_{mag}.pkl", "rb") as f:
            magmax_spline = pickle.load(f)
        splines.append(magmax_spline)
    if include_f_esc_const:
        for mag in abs_mag_list:
            with open(folder + f"f_esc_const_N_ion_magmin_33_magmax_{mag}.pkl", "rb") as f:
                magmax_spline = pickle.load(f)
            splines.append(magmax_spline)

elif muvmin_compare:
    splines = []
    mag_range = (-35, -25)
    abs_mag_list = list(range(abs(mag_range[1]), abs(mag_range[0]) + 1))[::-1]
    for mag in abs_mag_list:
        with open(folder + f"thesan_N_ion_magmin_{mag}_magmax_13.pkl", "rb") as f:
            magmin_spline = pickle.load(f)
        splines.append(magmin_spline)

else:
    splines = [thesan_spline, thesan_spline_high, thesan_spline_low,
        observational_spline, observational_spline_high, observational_spline_low]   

Q_initial = 0.0
z_initial, z_final = 20, 0
sols = []
for spl in splines:
    sol = solve_ivp(fun = lambda z, Q: dQ_dz(z, Q, spl),
                    t_span=[z_initial, z_final], y0=[Q_initial],
                    t_eval=np.linspace(z_initial, z_final, 20000))
    sols.append(sol)

thesan_finish = sols[0].t[np.where(sols[0].y[0] >= 1)[0][0]]
thesan_finish_high = sols[1].t[np.where(sols[1].y[0] >= 1)[0][0]]
thesan_finish_low = sols[2].t[np.where(sols[2].y[0] >= 1)[0][0]]
observational_finish = sols[3].t[np.where(sols[3].y[0] >= 1)[0][0]]
observational_finish_high = sols[4].t[np.where(sols[4].y[0] >= 1)[0][0]]
observational_finish_low = sols[5].t[np.where(sols[5].y[0] >= 1)[0][0]]
print("According to Thesan-Zoom: Reionisation completes at an average redshift of z =", thesan_finish)
print(f"According to Thesan-Zoom: Reionisation completes in the redshift range: {thesan_finish_low} < z < {thesan_finish_high}")
print("According to JWST observations: Reionisation completes at an average redshift of z =", observational_finish)
print(f"According to JWST observations: Reionisation completes in the redshift range: {observational_finish_low} < z < {observational_finish_high}")

# Extract ionisation fractions at z=8
z_target = 8.0
thesan_Q_at_8 = np.interp(z_target, sols[0].t[::-1], sols[0].y[0][::-1])
jwst_Q_at_8 = np.interp(z_target, sols[3].t[::-1], sols[3].y[0][::-1])
print(f"Ionisation fraction at z = {z_target}:")
print(f"  Thesan-Zoom: Q = {thesan_Q_at_8:.4f}")
print(f"  JWST: Q = {jwst_Q_at_8:.4f}")

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(figsize=[(8, 6), [(8, 8), (12, 8)][include_f_esc_const]][Q_plot])
cmap1 = plt.get_cmap('inferno')
cmap2 = plt.get_cmap('viridis')
colours2 = cmap2(np.linspace(0.2, 0.8, len(sols)))[::-1]

if Q_plot == True:

    if sigma_compare:
        colours = cmap1(np.linspace(0.2, 0.8, len(sols)))[::-1]
        for index in range(len(sols)):
            ax.plot(sols[index].t, sols[index].y[0], label=f'$\sigma_{{\dot{{n}}}}$ = {sigmas[index]}',
                    linewidth=2, c=colours[index])
            print(f"For sigma_n = {sigmas[index]}: Reionisation completes at z =",
                  sols[index].t[np.where(sols[index].y[0] >= 1)[0][0]])

    elif muvmax_compare:
        thesan_lines = []
        f_esc_const_lines = []
        num_mags = len(abs_mag_list)
        colours1 = cmap1(np.linspace(0.2, 0.8, num_mags))[::-1]
        colours2 = cmap2(np.linspace(0.2, 0.8, num_mags))[::-1]
        for index in range(num_mags):
            line = ax.plot(sols[index].t, sols[index].y[0], label=f'$M_\mathrm{{UV,max}}$ = -{abs_mag_list[index]}',
                           linewidth=2, c=colours1[index])[0]
            thesan_lines.append(line)
            print(f"For M_UV_max = -{abs_mag_list[index]}: Reionisation completes at z =",
                  sols[index].t[np.where(sols[index].y[0] >= 1)[0][0]])
            if include_f_esc_const:
                line = ax.plot(sols[num_mags + index].t, sols[num_mags + index].y[0], label=f'$M_\mathrm{{UV,max}}$ = -{abs_mag_list[index]}',
                               linewidth=2, c=colours2[index], linestyle='--')[0]
                f_esc_const_lines.append(line)

    elif muvmin_compare:
        colours = cmap1(np.linspace(0.2, 0.8, len(sols)))[::-1]
        for index in range(len(abs_mag_list)):
            ax.plot(sols[index].t, sols[index].y[0], label=f'$M_\mathrm{{UV,min}}$ = -{abs_mag_list[index]}',
                    linewidth=2, c=colours[index])
            print(f"For M_UV_min = -{abs_mag_list[index]}: Reionisation completes at z =",
                  sols[index].t[np.where(sols[index].y[0] >= 1)[0][0]])

    else:
        # plots the ionisation fraction with uncertainty range
        ax.plot(sols[0].t, sols[0].y[0], label=r'$\textsc{thesan-zoom}$-based', color='darkviolet', linewidth=2)
        ax.fill_between(sols[0].t, sols[1].y[0], sols[2].y[0], color='darkviolet', alpha=0.15)
        ax.plot(sols[3].t, sols[3].y[0], label='JWST-based', color='darkcyan', linestyle='--', linewidth=2)
        # ax.axvline(x=thesan_finish, color='red', linestyle='--')

        constraint_colours = {
            "Planck_2018":  "#4a0a14",  # Near-Black Red
            "Davies_2018":  "#7f1020",  # Dark Maroon
            "Wang_2020":    "#b11226",  # Deep Crimson
            "Jin_2023":     "#e6391c",  # Bright Scarlet
            "Zhu_2024":     "#ff5a1f",  # Red-Orange
            "Tang_2024":    "#ff8c1a",  # Strong Orange
            "Kageura_2025": "#ffb000",  # Golden Amber
        }

        # plots observational constraints on Q from Planck (2018)
        ax.errorbar(7.68, 0.5, xerr=[[0.79], [0.79]],
                    fmt='none', c=constraint_colours["Planck_2018"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(7.68, 0.5, marker='s', color=constraint_colours["Planck_2018"], s=75, edgecolors='black', label='Planck 2018', zorder=5)

        # plots observational constraints on Q from Davies et al. (2018)
        davies_z = np.array([7.0851, 7.5413])
        davies_Q = 1 - np.array([0.52, 0.67])
        davies_Q_err_low = np.array([0.25, 0.19])
        davies_Q_err_high = np.array([0.25, 0.23])
        ax.errorbar(davies_z, davies_Q, yerr=[davies_Q_err_low, davies_Q_err_high],
                    fmt='none', c=constraint_colours["Davies_2018"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(davies_z, davies_Q, marker='H', color=constraint_colours["Davies_2018"], s=75, edgecolors='black', label='Davies+2018', zorder=5)

        # plots observational constraint on Q from Wang et al. (2020)
        ax.errorbar(7.00, 1-0.70, yerr=[[0.23], [0.20]],
                    fmt='none', c=constraint_colours["Wang_2020"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(7.00, 1-0.70, marker='p', color=constraint_colours["Wang_2020"], s=75, edgecolors='black', label='Wang+2020', zorder=5)

        # plots observational constraints on Q from Jin et al. (2023)
        jin_z = np.array([5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7])
        jin_Q = 1 - np.array([0.09, 0.16, 0.28, 0.69, 0.79, 0.87, 0.94])
        jin_Q_err_low = np.array([0.08, 0.14, 0.08, 0.06, 0.04, 0.03, 0.06])
        jin_Q_err_high = np.array([0.08, 0.14, 0.08, 0.06, 0.04, 0.03, 0.09])
        ax.errorbar(jin_z, jin_Q, yerr=[jin_Q_err_low, jin_Q_err_high],
                    fmt='none', c=constraint_colours["Jin_2023"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(jin_z, jin_Q, marker='^', color=constraint_colours["Jin_2023"], s=75, edgecolors='black', label='Jin+2023', zorder=5)
        ax.plot(jin_z, jin_Q + jin_Q_err_high + 0.011, marker=6, color=constraint_colours["Jin_2023"], markersize=8, linestyle='None')

        # plots observational constraints on Q from Umeda et al. (2024)
        # umeda_z = np.array([5.7, 6.6, 7.0, 7.3])
        # umeda_Q = 1 - np.array([0.06, 0.15, 0.18, 0.75])
        # umeda_Q_err_low = np.array([0.12, 0.19, 0.14, 0.09])
        # umeda_Q_err_high = np.array([0.03, 0.14, 0.12, 0.13])
        # ax.errorbar(umeda_z, umeda_Q, yerr=[umeda_Q_err_low, umeda_Q_err_high],
        #              fmt='none', c='green', elinewidth=1.5, capsize=4, label='Umeda+2024', alpha=0.8, zorder=4)
        # ax.scatter(umeda_z, umeda_Q, marker='*', color='green', s=75, edgecolors='black', zorder=5)

        # plots observational constraints on Q from Zhu et al. (2024)
        zhu_z = np.array([5.8])
        zhu_z_low = np.array([5.4])
        zhu_z_high = np.array([6.2])
        zhu_Q = 1 - np.array([0.061])
        zhu_Q_err_low = np.array([0.039])
        zhu_Q_err_high = np.array([0.039])
        ax.errorbar(zhu_z, zhu_Q, xerr=[zhu_z - zhu_z_low, zhu_z_high - zhu_z], yerr=[zhu_Q_err_low, zhu_Q_err_high],
            fmt='none', c=constraint_colours["Zhu_2024"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(zhu_z, zhu_Q, marker='v', color=constraint_colours["Zhu_2024"], s=75, edgecolors='black', label='Zhu+2024', zorder=5)
        ax.plot(zhu_z, zhu_Q - zhu_Q_err_high - 0.011, marker=7, color=constraint_colours["Zhu_2024"], markersize=8, linestyle='None')


        # plots observational constraints on Q from Tang et al. (2024)
        tang_z = np.array([7.0, 8.8, 11.0])
        tang_z_low = np.array([6.0, 8.0, 10.0])
        tang_z_high = np.array([8.0, 10.0, 13.3])
        tang_Q = 1 - np.array([0.48, 0.81, 0.89])
        tang_Q_err_low = np.array([0.15, 0.12, 0.08])
        tang_Q_err_high = np.array([0.22, 0.24, 0.21])
        ax.errorbar(tang_z, tang_Q, yerr=[tang_Q_err_low, tang_Q_err_high], xerr=[tang_z - tang_z_low, tang_z_high - tang_z],
                    fmt='none', c=constraint_colours["Tang_2024"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(tang_z, tang_Q, marker='D', color=constraint_colours["Tang_2024"], s=75, edgecolors='black',  label='Tang+2024', zorder=5)

        # plots observational constraints on Q from Kageura et al. (2025)
        kageura_z = np.array([5.90, 6.96, 8.41, 11.00])
        kageura_z_low = np.array([5.50, 6.54, 7.51, 9.62])
        kageura_z_high = np.array([6.39, 7.49, 9.43, 14.18])
        kageura_Q = 1 - np.array([0.17, 0.63, 0.79, 0.88])
        kageura_Q_err_low = np.array([0.23, 0.18, 0.13, 0.11])
        kageura_Q_err_high = np.array([0.16, 0.28, 0.21, 0.13])
        ax.errorbar(kageura_z, kageura_Q, yerr=[kageura_Q_err_low, kageura_Q_err_high], xerr=[kageura_z - kageura_z_low, kageura_z_high - kageura_z],
                    fmt='none', color=constraint_colours["Kageura_2025"], elinewidth=1.5, capsize=4, alpha=0.8, zorder=4)
        ax.scatter(kageura_z, kageura_Q, marker='o', color=constraint_colours["Kageura_2025"], s=75, edgecolors='black', label='Kageura+2025', zorder=5)

    ax.set_ylim(0, 1)
    ax.set_xlim(4, 16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.grid(False)
    ax.set_xlabel('$z$')
    ax.set_ylabel('$Q_\mathrm{HII}$')

    if not include_f_esc_const:
        legend = plt.legend(fontsize=(20, 24)[sigma_compare], loc='upper right', bbox_to_anchor=(1.00, 1.00), frameon=False)
    else:
        header_thesan = ax.plot([], [], linestyle='none', linewidth=0, marker=None, label=' ')[0]
        header_text_thesan = ax.text(1.195, 0.94, r'$\textsc{thesan-zoom}$-based:', fontsize=16,
                                     transform=ax.transAxes, ha='center', va='center', zorder=5)
        header_f_esc_const_1 = ax.plot([], [], linestyle='none', linewidth=0, marker=None, label=' ')[0]
        header_f_esc_const_2 = ax.plot([], [], linestyle='none', linewidth=0, marker=None, label=' ')[0]
        header_text_f_esc_const = ax.text(1.195, 0.47, r'$f_\mathrm{esc} = 10$\%:', fontsize=16,
                                          transform=ax.transAxes, ha='center', va='center', zorder=5)
        handles = [header_thesan, *thesan_lines, header_f_esc_const_1, header_f_esc_const_2, *f_esc_const_lines]
        labels = [h.get_label() for h in handles]
        legend = ax.legend(handles, labels, alignment='center', handlelength=1.2,handletextpad=0.6, borderpad=1.4, labelspacing=0.4,
                           fontsize=16, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
        legend.set_zorder(4)
        for text in legend.get_texts():
                text.set_ha('center')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_boxstyle('Square')
        frame.set_alpha(0.8)
        plt.subplots_adjust(right=0.7)

elif sigma_compare == False and muvmax_compare == False and muvmin_compare == False:

    z, thesan_tau_z = tau(sols[0])
    _, thesan_tau_z_low = tau(sols[2])
    _, thesan_tau_z_high = tau(sols[1])
    _, obvs_tau_z = tau(sols[3])

    print("Thesan-Zoom Average Thompson optical depth tau =", thesan_tau_z[-1])
    print(f"Thesan-Zoom Thompson optical depth tau range: {thesan_tau_z_low[-1]} < tau < {thesan_tau_z_high[-1]}")

    planck_tau = 0.0544
    planck_tau_err_high = 0.0070
    planck_tau_err_low = 0.0081
    ax.fill_between(z, planck_tau - 2*planck_tau_err_low, planck_tau + 2*planck_tau_err_high,
                    color='#6E6E6E', edgecolor='#3E3E3E', alpha=0.15, label='Planck+2018', zorder=2)

    # heinrich_tau = 0.0619
    # heinrich_tau_err_high = 0.0056
    # heinrich_tau_err_low = 0.0068
    # ax.fill_between(z, heinrich_tau - 2*heinrich_tau_err_low, heinrich_tau + 2*heinrich_tau_err_high,
    #                 color='#2A9D8F', edgecolor='#146A5A', alpha=0.2, label='Heinrich+2021', zorder=2)

    ax.plot(z, thesan_tau_z, label=r'$\textsc{thesan-zoom}$-based', linewidth=3, color='darkviolet', zorder=3)
    ax.fill_between(z, thesan_tau_z_low, thesan_tau_z_high, color='darkviolet', alpha=0.15, zorder=3)
    ax.plot(z, obvs_tau_z, label=r'JWST-based', linewidth=3, color='darkcyan', linestyle='--', zorder=3)

    ax.set_ylim(0, 0.08)
    ax.set_xlim(4, 16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.grid(False)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$\tau$')
    legend = plt.legend(fontsize=20, loc='lower right', bbox_to_anchor=(0.975, 0.025), frameon=False)

plt.tight_layout()
mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()