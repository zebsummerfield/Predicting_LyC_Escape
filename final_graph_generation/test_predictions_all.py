import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

# True to use the dusty Thesan-Zoom catalogue, False for dust-free catalogue
dusty = True

def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

folder = "final_rf_model/"
dusty_str = ['','_dusty'][dusty]
file1 = folder + f'f_esc_rf_final{dusty_str}_test_train.json'
file2 = folder + f'n_esc_rf_final{dusty_str}_test_train.json'
file3 = folder + f'f_esc_rf_observational_charlotte{dusty_str}_test_train.json'
file4 = folder + f'n_esc_rf_observational_charlotte{dusty_str}_test_train.json' 
files = [file1, file2, file3, file4]
model_strs = ['Model A', 'Model B', 'Model C', 'Model D']
model_strs_2 = ['Best $f_\mathrm{esc}$',
                'Best $\dot{N}_\mathrm{ion,esc}$',
                'Observational $f_\mathrm{esc}$', 
                'Observational $\dot{N}_\mathrm{ion,esc}$']
model_strs_3 = 'Predictor'
colours = [mpl.colormaps['Purples'](1.0),
           mpl.colormaps['Reds'](1.0),
           mpl.colormaps['Blues'](1.0),
           mpl.colormaps['Oranges'](1.0)]
cmaps = [truncate_colormap(plt.cm.Purples, 0.1, 1.0),
         truncate_colormap(plt.cm.Reds, 0.1, 1.0),
         truncate_colormap(plt.cm.Blues, 0.1, 1.0),
         truncate_colormap(plt.cm.Oranges, 0.1, 1.0)]

# True if histogram bin colour is logarithmic
log = False

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 24})
box_width = 0.2
fig_ratio = 3
plot_height = 8
fig = plt.figure(figsize=(plot_height*fig_ratio, plot_height))
# Create subplots with explicit positions - this gives us total control
# [left, bottom, width, height] - all values are fractions of figure size
ax1 = fig.add_axes([0.025, 0.30, box_width, box_width * fig_ratio])  # prediction f_esc Model A
ax2 = fig.add_axes([0.275, 0.30, box_width, box_width * fig_ratio])   # prediction f_esc Model B
ax3 = fig.add_axes([0.525, 0.30, box_width, box_width * fig_ratio])   # prediction f_esc Model C
ax4 = fig.add_axes([0.775, 0.30, box_width, box_width * fig_ratio])    # prediction f_esc Model D
ax5 = fig.add_axes([0.025, 0.08, box_width, box_width * fig_ratio / 3])   # MAE f_esc Model A
ax6 = fig.add_axes([0.275, 0.08, box_width, box_width * fig_ratio / 3])    # MAE f_esc Model B
ax7 = fig.add_axes([0.525, 0.08, box_width, box_width * fig_ratio / 3])   # MAE f_esc Model C
ax8 = fig.add_axes([0.775, 0.08, box_width, box_width * fig_ratio / 3])    # MAE f_esc Model D
axes = np.array([[ax1, ax2, ax3, ax4],
                 [ax5, ax6, ax7, ax8]])
cbar_ax1 = fig.add_axes([0.025, 0.92, box_width, 0.02]) # colorbar axis for model A plot
cbar_ax2 = fig.add_axes([0.275, 0.92, box_width, 0.02]) # colorbar axis for model B plot
cbar_ax3 = fig.add_axes([0.525, 0.92, box_width, 0.02]) # colorbar axis for model C plot
cbar_ax4 = fig.add_axes([0.775, 0.92, box_width, 0.02]) # colorbar axis for model D plot
cbars = [cbar_ax1, cbar_ax2, cbar_ax3, cbar_ax4]
hists = []

x_strs = ["$\mathrm{log}_{10}(f_\mathrm{esc})$ Predicted", "$\mathrm{log}_{10}(\dot{N}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$ Predicted"]
y_strs = ["$\mathrm{log}_{10}(f_\mathrm{esc})$", "$\mathrm{log}_{10}(\dot{N}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$"]

for m_i in range(len(model_strs)):
    x_str = x_strs[m_i % 2]
    y_str = y_strs[m_i % 2]

    with open(files[m_i], 'r') as json_data:
        print(f"\n{'='*100}\n {files[m_i]} \n{'='*100}")
        data = json.load(json_data)
        test = np.array(data['f_esc_test'])
        test_pred = np.array(data['f_esc_test_pred'])
        train = np.array(data['f_esc_train'])
        train_pred = np.array(data['f_esc_train_pred'])

    # calculate errors on the test and train data
    test_mae = mean_absolute_error(test, test_pred)
    test_mse = mean_squared_error(test, test_pred)
    test_rmse = root_mean_squared_error(test, test_pred)
    test_r = np.corrcoef(test, test_pred)[0, 1]
    print(f"Test Mean Absolute Error: {test_mae}")
    print(f"Test Mean Squared Error: {test_mse}")
    print(f"Test Root Mean Squared Error: {test_rmse}")
    print(f"Test Correlation Coefficient: {test_r}")
    train_mae = mean_absolute_error(train, train_pred)
    train_mse = mean_squared_error(train, train_pred)
    train_rmse = root_mean_squared_error(train, train_pred)
    train_r = np.corrcoef(train, train_pred)[0, 1]
    print(f"Train Mean Absolute Error: {train_mae}")
    print(f"Train Mean Squared Error: {train_mse}")
    print(f"Train Root Mean Squared Error: {train_rmse}")
    print(f"Train Correlation Coefficient: {train_r}")
    print(f"Maximum Test Prediction: {max(test_pred)}")
    print(f"Minimum Test Prediction: {min(test_pred)}")

    x_min, x_max = [(-4, 0), (46, 54)][m_i % 2]
    nbins = (75, 45)[dusty]
    x_bins = np.linspace(x_min, x_max, nbins+1)
    
    axes[0, m_i].set_xlim(x_min, x_max)
    axes[0, m_i].set_ylim(x_min, x_max)
    axes[1, m_i].set_xlim(x_min, x_max)
    # hide x labels for top plots to improve appearance
    plt.setp(axes[0, m_i].get_xticklabels(), visible=False)

    # plots a 2d histogram of predicted test target against test target,
    # where the number of galaxies in a bin dictates it's colour
    hist, xedges, yedges = np.histogram2d(test_pred, test, bins=[x_bins, x_bins])
    hist = hist.T
    hist = np.ma.masked_where(hist == 0, hist)
    if log:
        hist = np.log10(hist)
    # cmap.set_bad(color='#d3d3d3')
    h1 = axes[0, m_i].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                              origin='lower', aspect='auto', cmap=cmaps[m_i], interpolation='nearest', zorder=1)
    hists.append(h1)
    # ensure ylabel doesn't get cut off
    axes[0, m_i].set_ylabel(y_str)
    axes[0, m_i].yaxis.set_label_coords(-0.1, 0.5)
    cbars[m_i].xaxis.set_label_coords(0.5, 3.5)

    # seperates the galaxies into bins of predicted test target with each containing equal numbers of galaxies, 
    # then plots the median of predicted test target against the median of test target for each bin
    nbins = [25, 15][dusty]
    bins = np.quantile(test_pred, np.linspace(0, 1, nbins + 1))
    bin_indices = np.digitize(test_pred, bins)
    test_pred_medians = np.zeros(nbins) 
    test_medians = np.zeros(nbins)
    test_pred_mae  = np.zeros(nbins)
    test_pred_mse = np.zeros(nbins) 
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        test_pred_medians[i-1] = np.median(test_pred[bin_mask])
        test_medians[i-1] = np.median(test[bin_mask])
        test_pred_mae[i-1] = mean_absolute_error(test[bin_mask], test_pred[bin_mask])
        test_pred_mse[i-1] = mean_squared_error(test[bin_mask], test_pred[bin_mask])
    axes[0, m_i].plot(test_pred_medians, test_medians, c='red', linewidth=3, alpha=0.8, zorder=3)
    # axes[0, m_i].fill_between(test_pred_medians, test_medians - test_pred_mae, test_medians + test_pred_mae,
    #                           color='r', alpha=0.2, label="16th-84th percentile", zorder=5)

    cut = [ -2, 50][m_i % 2]
    print(f"Mean Absolute Error for {('f_esc', 'n_esc')[m_i % 2]} > {cut}: {mean_absolute_error(test[test_pred > cut], test_pred[test_pred > cut])}")

    # plots the line of y = x
    axes[0, m_i].plot((x_min, x_max), (x_min, x_max), c='black', linewidth=1.5, alpha=0.6, zorder=2)

    bars = True
    if not bars:
        # plots the mean absolute error against the median predicted test target for each bin
        axes[1, m_i].plot(test_pred_medians, test_pred_mae, c=colours[m_i], linewidth=3, alpha=0.9)
    else:
        # plots the mean absolute error against the median predicted test target for each bin as a bar chart
        bin_centres = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        axes[1, m_i].bar(bin_centres, test_pred_mae, width=bin_widths,
                          color=colours[m_i], alpha=0.7, edgecolor='black')
        
    axes[0, m_i].text(0.05, 0.95, model_strs[m_i] + ' - \n' + model_strs_2[m_i] + '\n' + model_strs_3, 
                      ha='left', va='top', transform=axes[0, m_i].transAxes)
    
    axes[1, m_i].set_ylim(0.250, 0.650)
    axes[1, m_i].set_xlabel(x_str)
    axes[1, m_i].set_ylabel('MAE [dex]')
    axes[1, m_i].yaxis.set_label_coords(-0.1, 0.5)
    axes[1, m_i].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[1, m_i].yaxis.set_major_locator(MaxNLocator(nbins=2))
    axes[0, m_i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    fig.align_ylabels([axes[0, m_i], axes[1, m_i]])

    # axes[0, ax_i].grid(True, alpha=0.4, linestyle='--')
    # axes[1, ax_i].grid(True, alpha=0.8, linestyle='--')
    for ax in axes[:,m_i]:
        ax.grid(False)
        # ax.set_axisbelow(True)
        # for line in ax.get_xgridlines() + ax.get_ygridlines():
        #     line.set_zorder(0)


cbar1 = fig.colorbar(hists[0], cax=cbar_ax1, orientation='horizontal')
cbar2 = fig.colorbar(hists[1], cax=cbar_ax2, orientation='horizontal')
cbar3 = fig.colorbar(hists[2], cax=cbar_ax3, orientation='horizontal')
cbar4 = fig.colorbar(hists[3], cax=cbar_ax4, orientation='horizontal')
cbars = [cbar1, cbar2, cbar3, cbar4]
if log:
    colour_label = "$\mathrm{log}_{10}(N_\mathrm{gal})$"
else:
    colour_label = "$N_\mathrm{gal}$"
for cbar in cbars:
    cbar.set_label(colour_label)
    # Move ticks and labels to the top of the colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.labelpad = 100
    label = cbar.ax.xaxis.get_label()
    x, y = label.get_position()
    label.set_position((x, y + 0.5))    

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.pdf", bbox_inches='tight', dpi=500)
plt.show()