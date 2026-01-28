
import matplotlib
import matplotlib.pyplot as plt

x = [-18.160452, -16.67829, -15.799621, -15.135045, -14.606108,
     -14.160326, -13.744327, -13.28509, -12.760942, -11.9468975]
y = [0.5701199059671632, 0.5035665397553862, 0.5659262813977876, 0.6465841170902499, 0.6805833655723811,
     0.7410335356623795, 0.7316941679021577, 0.8092791924905124, 0.8226107174978412, 0.8618559317730994]
y = [0.82325226, 0.54033775, 0.58235153, 0.60632828, 0.60632828,
     0.64126038, 0.67182114, 0.71882866, 0.74310898, 0.71705704]
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 1, figsize=((8, 8)))

ax.plot(x, y, marker='o', linestyle='-', markersize=8, markeredgecolor='black', markerfacecolor='red', alpha=0.8)
ax.set_xlabel("$M_\mathrm{UV}$")
ax.set_ylabel("$\sigma_{\dot{n}}$")
ax.set_xlim(-20, -10)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()