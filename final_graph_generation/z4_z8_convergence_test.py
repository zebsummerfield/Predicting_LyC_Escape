import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from functions import *

# 0 for f_esc, 1 for n_esc
f_or_n = 0

# True if model is generated to predict for an observational catalogue 
obvs = False

# True to use the dusty Thesan-Zoom catalogue, False for dust-free catalogue
dusty = False

folder = "final_graph_generation/"
file = ['cat.hdf5', 'cat_dusttestszeb_fdust.hdf5'][dusty]

# Load data for both zoom regions
zoom_data = {}
for zoom in ['z4', 'z8']:
    keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, dusty=dusty, eps=True, zoom=zoom, add_vars=['redshift_full'])
    zoom_data[zoom] = {
        'keys': keys,
        'log_vars': log_vars,
        'log_f_esc': log_f_esc,
        'log_n_esc': log_n_esc,
        'log_x': log_vars[6],  # index of M_UV
        'z': 10**log_vars[-1]
    }

print(len(zoom_data['z4']['log_f_esc']))
print(zoom_data['z4']['keys'])

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 15})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
## Create figure with GridSpec to have better control over subplots and colorbar
# fig = plt.figure(figsize=(12, 5))
# gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.2)  # The last slice is for colorbar
# axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
# cbar_ax = fig.add_subplot(gs[0, 3])  # Dedicated axes for colorbar

# Define colors and labels for each zoom
zoom_config = {'z4': {'color': 'r', 'label': 'z4'}, 'z8': {'color': 'b', 'label': 'z8'}}

for zoom in ['z8', 'z4']:
    keys = zoom_data[zoom]['keys']
    log_f_esc = zoom_data[zoom]['log_f_esc']
    log_n_esc = zoom_data[zoom]['log_n_esc']
    log_x = zoom_data[zoom]['log_x']
    color = zoom_config[zoom]['color']
    zoom_label = zoom_config[zoom]['label']
    
    # Plot 2d histogram for z4 (before filtering)
    for ax_i in range(len(axes)):
        log_y = [log_f_esc, log_n_esc][ax_i]
        f_or_n_str = ['$\mathrm{log}_{10}(f_{\mathrm{esc}}) \; - \; \mathrm{0\% \, dust}$',
                      '$\mathrm{log}_{10}(\dot{N}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}]) \; - \; \mathrm{0\% \, dust}$'][ax_i]
        y_limits = ((-5, 0), (46, 55))[ax_i]

        # plots a 2d histogram of log_x against log_y where the number of galaxies in a bin dictates it's colour
        # only for z4 to set the heatmap background (before nearest neighbor filtering)
        if zoom == 'z4':
            nbins = (100, 75)[dusty]
            hist, xedges, yedges = np.histogram2d(log_x, log_y, bins=nbins, range=((min(log_x), max(log_x)), y_limits))
            hist = hist.T
            hist = np.log10(hist)
            hist[hist == -np.inf] = 0
            h1 = axes[ax_i].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                            origin='lower', aspect='auto', cmap='viridis', interpolation='nearest', vmin=0)
            axes[ax_i].set_xlabel("$M_{\mathrm{UV}}$")
            axes[ax_i].set_ylabel(f_or_n_str)
    
    # Apply nearest neighbor matching for z4 to match z8 sample
    if zoom == 'z4':
        # Get stellar mass (index 2) and redshift (index -1) for matching
        z4_stellar_mass = zoom_data['z4']['log_vars'][2]  # star_mass
        z4_redshift = zoom_data['z4']['log_vars'][-1]  # redshift_full
        z8_stellar_mass = zoom_data['z8']['log_vars'][2]
        z8_redshift = zoom_data['z8']['log_vars'][-1]
        
        # Standardize features so stellar mass and redshift contribute comparably
        z4_features_raw = np.column_stack([z4_stellar_mass, z4_redshift])
        z8_features_raw = np.column_stack([z8_stellar_mass, z8_redshift])
        scaler = StandardScaler().fit(z4_features_raw)
        z4_features = scaler.transform(z4_features_raw)
        z8_features = scaler.transform(z8_features_raw)
        
        # Build NearestNeighbors model on z4 data
        nbrs = NearestNeighbors(n_neighbors=10000, algorithm='ball_tree').fit(z4_features)
        distances, indices = nbrs.kneighbors(z8_features)
        used = set()
        matched_indices = []
        for neighbours in indices:
            match = next(i for i in neighbours if i not in used)
            matched_indices.append(match)
            used.add(match)
        matched_indices = np.array(matched_indices)
        
        # Create selection mask for z4 galaxies matching z8
        selection_mask = np.zeros(len(log_x), dtype=bool)
        selection_mask[matched_indices] = True
        
        # Apply selection to z4 data
        log_x = log_x[selection_mask]
        log_f_esc = log_f_esc[selection_mask]
        log_n_esc = log_n_esc[selection_mask]

        dm = z4_stellar_mass[matched_indices] - z8_stellar_mass
        dz = z4_redshift[matched_indices] - z8_redshift
        print("Mass difference:", np.median(abs(dm)), np.max(abs(dm)))
        print("Redshift difference:", np.median(abs(dz)), np.max(abs(dz)))
        
        print(f"Z4 matched sample size: {len(log_x)} (selected from {len(selection_mask)} original)")
    
    # Plot medians and statistics for binned data
    for ax_i in range(len(axes)):
        log_y = [log_f_esc, log_n_esc][ax_i]
        f_or_n_str = ['$\mathrm{log}_{10}(f_{\mathrm{esc}}) \; - \; \mathrm{0\% \, dust}$',
                      '$\mathrm{log}_{10}(\dot{N}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}]) \; - \; \mathrm{0\% \, dust}$'][ax_i]
        y_limits = ((-5, 0), (46, 55))[ax_i]

        # seperates the galaxies into bins of variable log_x with each containing equal numbers of galaxies 
        nbins = 25
        bins = np.quantile(log_x, np.linspace(0, 1, nbins + 1))
        bin_indices = np.digitize(log_x, bins)
        x_medians, y_medians = ([], [])
        y_16th, y_84th = ([], [])
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            x_medians.append(np.median(log_x[bin_mask]))
            y_medians.append(np.median(log_y[bin_mask]))
            y_16th.append(np.percentile(log_y[bin_mask], 16))
            y_84th.append(np.percentile(log_y[bin_mask], 84))

        # plots the median of log_x against the median of log_y for each bin as well as the 1 sigma range
        median_plot = axes[ax_i].plot(x_medians, y_medians, c=color, linewidth=3, alpha=0.8, label=f"median $f_{{esc}}$ ({zoom_label})", zorder=4)
        axes[ax_i].fill_between(x_medians, y_16th, y_84th, color=color, alpha=0.2, label=f"P16 -- P84 ({zoom_label})", zorder=4)

        # calculates the correlation coefficient and mean 1-sigma range
        selected_galaxies = np.where(log_x < -13)[0]
        corrcoef = np.corrcoef(log_x[selected_galaxies], log_y[selected_galaxies])[0, 1]
        print(f"{zoom.upper()} - Correlation coefficient between M_UV and {f_or_n_str}: {corrcoef:.3f}")
        mean_sigma = np.mean(np.array(y_84th) - np.array(y_16th)) / 2
        print(f"{zoom.upper()} - Mean 1-sigma range for {f_or_n_str}: {mean_sigma:.3f} dex")

        if zoom == 'z4':
            axes[ax_i].set_xlim(-21, -13)
            axes[ax_i].set_ylim(y_limits)
            axes[ax_i].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            axes[ax_i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            axes[ax_i].set_box_aspect(1)
            axes[ax_i].tick_params(left=False, right=False, top=False, bottom=False)
            axes[ax_i].minorticks_off()
            axes[ax_i].grid(False)
            #axes[ax_i].grid(True, alpha=0.6, linestyle='--')
            axes[ax_i].legend(fontsize=10, loc='lower left', frameon=False)

cbar = fig.colorbar(h1, label="$\mathrm{log}_{10}(N_\mathrm{gal})$",
                    ax=axes, fraction=0.046)
fig.canvas.draw()   # <--- force creation of cbar.solids
if hasattr(cbar, 'solids') and cbar.solids is not None:
    # make patch edges the same colour as the face (removes seams)
    try:
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_linewidth(0.0)
        cbar.solids.set_antialiased(False)
    except Exception:
        # some mpl versions may behave slightly differently; this try/except is safe
        pass

mpl.rcParams['figure.dpi'] = 500
fig.savefig(folder + "report_graphs/report_graph.pdf", bbox_inches='tight', dpi=500)
plt.show()