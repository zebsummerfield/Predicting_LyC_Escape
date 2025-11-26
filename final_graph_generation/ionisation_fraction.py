import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib as mpl
import numpy as np
from sklearn.metrics import mean_absolute_error
from functions import *
import numpy as np
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.integrate import solve_ivp 

Y = 0.24668
X = 1 - Y
chi = Y / (4 * X)
alpha_B = 2.5775e-13 # cm^3 s^-1 at T=1e4 K
n_H_0 = 1.8786e-7 # cm^-3
omega_m = 0.3089
omega_lambda = 0.6911
H_0 = 2.195e-18 # s^-1

folder = "final_graph_generation/"
with open(folder + "uv_N_ion_spline_sch.pkl", "rb") as f:
    spline = pickle.load(f)
with open(folder + "uv_N_ion_spline_high_sch.pkl", "rb") as f:
    spline_high = pickle.load(f)
with open(folder + "uv_N_ion_spline_low_sch.pkl", "rb") as f:
    spline_low = pickle.load(f)

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

# solves the ODE for ionisation fraction as a function of redshift
Q_initial = 0.0
z_initial, z_final = 20, 0
sols = []
for spl in [spline, spline_high, spline_low]:
    sol = solve_ivp(fun = lambda z, Q: dQ_dz(z, Q, spl),
                    t_span=[z_initial, z_final], y0=[Q_initial],
                    t_eval=np.linspace(z_initial, z_final, 1000))
    sols.append(sol)

finish = sols[0].t[np.where(sols[0].y[0] >= 1)[0][0]]
finish_low = sols[2].t[np.where(sols[2].y[0] >= 1)[0][0]]
finish_high = sols[1].t[np.where(sols[1].y[0] >= 1)[0][0]]
print("Reionisation completes at an average redshift of z =", finish)
print(f"Reionisation redshift completion range: {finish_low} < z < {finish_high}")

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(8,8))

# plots the ionisation fraction with uncertainty range
ax.plot(sols[0].t, sols[0].y[0], label='$Q_{HII}$', color='Indigo')
ax.fill_between(sols[0].t, sols[1].y[0], sols[2].y[0], color='Indigo', alpha=0.25, label='$Q_{HII}$ Uncertainty')
# ax.axvline(x=finish, color='red', linestyle='--')

constraint_colours = {
    "Planck_2018": "#e377c2",     # Magenta
    "Davies_2018": "#1f78b4",     # Medium blue
    "Wang_2020": "#33a7c2",       # Blue-teal
    "Jin_2023": "#1b7837",        # Green
    "Tang_2024": "#e66101",       # Orange
    "Kageura_2025": "#a6761d",    # Brown-orange
}

# plots observational constraints on Q from Planck (2018)
ax.errorbar(7.68, 0.5, xerr=[[0.79], [0.79]],
            fmt='s', color=constraint_colours["Planck_2018"], ecolor=constraint_colours["Planck_2018"],
            markersize=7.5, mec='black', mew=1, capsize=4, label='Planck (2018)')

# plots observational constraints on Q from Davies (2018)
davies_z = np.array([7.0851, 7.5413])
davies_Q = 1 - np.array([0.52, 0.67])
davies_Q_err_low = np.array([0.25, 0.19])
davies_Q_err_high = np.array([0.25, 0.23])
ax.errorbar(davies_z, davies_Q, yerr=[davies_Q_err_low, davies_Q_err_high],
            fmt='o', color=constraint_colours["Davies_2018"], ecolor=constraint_colours["Davies_2018"],
            markersize=7.5, mec='black', mew=1, capsize=4, label='Davies (2018)')

# plots observational constraint on Q from Wang (2020)
ax.errorbar(7.00, 1-0.70, yerr=[[0.23], [0.20]],
            fmt='o', color=constraint_colours["Wang_2020"], ecolor=constraint_colours["Wang_2020"],
            markersize=7.5, mec='black', mew=1, capsize=4, label='Wang (2020)')

# plots observational constraints on Q from Jin (2023)
jin_z = np.array([5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7])
jin_Q = 1 - np.array([0.09, 0.16, 0.28, 0.69, 0.79, 0.87, 0.94])
jin_Q_err_low = np.array([0.08, 0.14, 0.08, 0.06, 0.04, 0.03, 0.06])
jin_Q_err_high = np.array([0.08, 0.14, 0.08, 0.06, 0.04, 0.03, 0.09])
ax.errorbar(jin_z, jin_Q, yerr=[jin_Q_err_low, jin_Q_err_high],
            fmt='^', color=constraint_colours["Jin_2023"], ecolor=constraint_colours["Jin_2023"],
            markersize=7.5, mec='black', mew=1,
            capsize=4, label='Jin (2023)')
ax.plot(jin_z, jin_Q + jin_Q_err_high + 0.0125, marker=6, color=constraint_colours["Jin_2023"], markersize=7.5, linestyle='None')

# # plots observational constraints on Q from Umeda (2024)
# umeda_z = np.array([5.7, 6.6, 7.0, 7.3])
# umeda_Q = 1 - np.array([0.06, 0.15, 0.18, 0.75])
# umeda_Q_err_low = np.array([0.12, 0.19, 0.14, 0.09])
# umeda_Q_err_high = np.array([0.03, 0.14, 0.12, 0.13])
# ax.errorbar(umeda_z, umeda_Q, yerr=[umeda_Q_err_low, umeda_Q_err_high],
#              fmt='o', color='green', markersize=7.5, ecolor='green', mec='black', mew=1,
#              capsize=4, label='Umeda (2024)')

# plots observational constraints on Q from Tang (2024)
tang_z = np.array([7.0, 8.8, 11.0])
tang_z_low = np.array([6.0, 8.0, 10.0])
tang_z_high = np.array([8.0, 10.0, 13.3])
tang_Q = 1 - np.array([0.48, 0.81, 0.89])
tang_Q_err_low = np.array([0.15, 0.12, 0.08])
tang_Q_err_high = np.array([0.22, 0.24, 0.21])
ax.errorbar(tang_z, tang_Q, yerr=[tang_Q_err_low, tang_Q_err_high],
            xerr=[tang_z - tang_z_low, tang_z_high - tang_z],
            fmt='o', color=constraint_colours["Tang_2024"], ecolor=constraint_colours["Tang_2024"],
            markersize=7.5, mec='black', mew=1, capsize=4, label='Tang (2024)')

# plots observational constraints on Q from Kageura (2025)
kageura_z = np.array([5.90, 6.96, 8.41, 11.00])
kageura_z_low = np.array([5.50, 6.54, 7.51, 9.62])
kageura_z_high = np.array([6.39, 7.49, 9.43, 14.18])
kageura_Q = 1 - np.array([0.17, 0.63, 0.79, 0.88])
kageura_Q_err_low = np.array([0.23, 0.18, 0.13, 0.11])
kageura_Q_err_high = np.array([0.16, 0.28, 0.21, 0.13])
ax.errorbar(kageura_z, kageura_Q, yerr=[kageura_Q_err_low, kageura_Q_err_high],
            xerr=[kageura_z - kageura_z_low, kageura_z_high - kageura_z],
            fmt='o', color=constraint_colours["Kageura_2025"], ecolor=constraint_colours["Kageura_2025"],
            markersize=7.5, mec='black', mew=1, capsize=4, label='Kageura (2025)')

ax.set_ylim(0, 1)
ax.set_xlim(3, 12)
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
ax.grid(False)
ax.set_xlabel('$z$')
ax.set_ylabel('$Q_\mathrm{HII}$')
legend = plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=False)
plt.show()