import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
import numpy as np
from astropy.io import fits
from scipy import odr
import csv
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import UnivariateSpline
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
with open(folder + "uv_N_ion_spline.pkl", "rb") as f:
    spline = pickle.load(f)
with open(folder + "uv_N_ion_spline_high.pkl", "rb") as f:
    spline_high = pickle.load(f)
with open(folder + "uv_N_ion_spline_low.pkl", "rb") as f:
    spline_low = pickle.load(f)

def N_ion(z,spl):
    return 10**spl(z) * (1/3.086e24)**3  # convert from Mpc^-3 to cm^-3

def n_H(z):
    return n_H_0 * (1 + z)**3

def t_rec(z):
    C = 9.25 - 7.21 * np.log10(1 + z)
    return 1 / ((1 + chi) * alpha_B * n_H(z) * C)

def H(z):
    return H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def dQ_dz(z, Q, spl):
    return - (1 / (H(z) * (1 + z))) * ((N_ion(z, spl) / n_H_0) - (Q / t_rec(z)))

Q_initial = 0.0
z_initial, z_final = 20, 0
sols = []
for spl in [spline, spline_high, spline_low]:
    sol = solve_ivp(fun = lambda z, Q: dQ_dz(z, Q, spl),
                    t_span=[z_initial, z_final], y0=[Q_initial],
                    t_eval=np.linspace(z_initial, z_final, 10000))
    sols.append(sol)

finish = sols[0].t[np.where(sols[0].y[0] >= 1)[0][0]]
print("Reionization complete at z =", finish)

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
plt.figure(figsize=(16,8))
plt.plot(sols[0].t, sols[0].y[0], label='$Q_{HII}$', color='blue')
plt.fill_between(sols[0].t, sols[1].y[0], sols[2].y[0], color='blue', alpha=0.3, label='Uncertainty Range')
plt.axvline(x=finish, color='red', linestyle='--', label='Reionization Completion')
plt.ylim(0, 1)
plt.xlim(z_final, z_initial)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
plt.grid(False)
plt.xlabel('$z$')
plt.ylabel('$Q_\mathrm{HII}$')
plt.show()