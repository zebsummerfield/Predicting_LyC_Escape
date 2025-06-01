# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:04:44 2021

@author: Gabe
"""

import os
import numpy as np
import scipy.stats as ss
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import scipy.optimize as so
import scipy.odr as sodr
import pandas as pd
import seaborn as sns
import pingouin as pg
import random
from cycler import cycler
from sklearn.utils import resample
from astropy.cosmology import FlatLambdaCDM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from lmfit import Model
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.colorbar as clbar

os.chdir('/Users/') # set cwd !!
path=os.getcwd()


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def make_contours(x,y, nbins=50, nlevels=None) :

    density_points, x_edges, y_edges = np.histogram2d(x, y, bins=(nbins,nbins), normed=True)
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
    levels = [zero_sigma1, one_sigma, two_sigma,three_sigma][::-1] #you can select here which levels you are interested in
    if nlevels is not None :
     levels = levels[nlevels:]
    
    return [X,Y, pdf,levels]

    
def rf_tuning(tab):
    """
    Function to perfom hyper-parameter tuning
    input astropy Table dataframe 
    """
    
    # read in different physics parameters we want to know the importances of
    M_s_list=tab['Mass_tot']
    sfr_list=tab['SFR_tot']
    metal_list=tab['12+log(O/H)_n2_r3']-8.69
    sfr_d_list=tab['SFR_D4000_inc_cut']
    random_list=np.random.random_sample(size=len(M_s_list))
    vd=np.log10(tab['SIGMA_BALMER']/100)
    
    # read in the variable we want to find importances relative to
    H_rat=tab['Balmer_dec']

    X=np.array([M_s_list, sfr_list, metal_list, sfr_d_list, vd, random_list]).T
    y=H_rat

    grid=np.arange(1, 300, 5) # grid of hyper-parameters want to iterate over

    ims=np.zeros([len(grid), len(X[0])]) # initialise empty list to input the importances
    MAEtes=np.zeros(len(grid)) # initialise mean average error for the test sample
    MSEtes=np.zeros(len(grid)) # initialise mean square error for the test sample
    MAEtrs=np.zeros(len(grid)) # initialise mean average error for the train sample
    MSEtrs=np.zeros(len(grid)) # initialise mean square error for the train sample
    
    n=5 # repeat this tuning process 5 times
        
    for j in range(n):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=j)
        # split the galaxies into test and train, 50:50 split due large sample
        
        for i in range(len(grid)):
        
            # initialise random forest regressor
            # set random state, so that each iteration is different but overall tuning is repeatable
            # min_samples_leaf only hyperparameter varied
            # n_jobs = -1 uses all of the cores in your machine
            regr = RandomForestRegressor(random_state=(i+1)*(j+1), min_samples_leaf=grid[i], n_jobs=-1)
            
            # fit the regressor to the training data
            regr.fit(X_train, y_train)
            
            # adds the importance of each parameter to ims array, /n to get average at end
            ims[i]+=regr.feature_importances_/n
            
            # use regressor to predict value of H_rat given the test data
            y_pred=regr.predict(X_test)
            
            # the following calculate the error in the regressor in predicting the test data,
            # put average in list from above
            MAEtes[i]+= mean_absolute_error(y_test, y_pred)/n
            MSEtes[i]+= mean_squared_error(y_test, y_pred)/n
            
            # now predict using the train data to check for overfitting
            y_pred_tr=regr.predict(X_train)
            
            # error in predicting the value of H_rat given the training data
            MAEtrs[i]+= mean_absolute_error(y_train, y_pred_tr)/n
            MSEtrs[i]+= mean_squared_error(y_train, y_pred_tr)/n
            
        print('Repetition ', j+1, ' out of ', n, ' done!')
    
    # percentage difference between the regressor performance using the test and train samples
    pdiff_MAE= abs(MAEtes-MAEtrs)/MAEtes
    pdiff_MSE= abs(MSEtes-MSEtrs)/MSEtes
    
    # this is the % threshold desired
    thresh=0.02
    
    # grid indices where MAE % difference is below the threshold
    indicesMAE=np.where(pdiff_MAE<thresh)
    
    # initialise figure to show result of hypertuning
    fig, ax = plt.subplots(figsize=(7,5))
    ax.grid(True)
    
    ax.plot(grid, MAEtes, label='Test')
    ax.plot(grid, MAEtrs, label='Train')
    ax.axvline(grid[indicesMAE[0][0]], linestyle='dashdot', label='MAE 2% threshold')
    ax.set_xlabel('Minimum Number of Samples on Final Leaf')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title(str_title)

    ax.set_xlim([0,300])
    ax.set_ylim([0.11,0.4])
    ax.legend()
    ax.tick_params(which='both',bottom=True,top=True,right=True)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(.01))
        
    fig_path=path+'/final_figs_paper/revised_plots/RF_tun_.png'
    fig.savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)

    # print results, use the value of min_samples_leaf at which 1% threshold reached in next function
    print('Minimum of MAE on test sample is: ', MAEtes.min(), ' at minimum number of samples: ', grid[np.where(MAEtes==MAEtes.min())[0][0]])
    print('1% threshold on difference in MAE has MAE of: ', MAEtes[indicesMAE[0][0]], ' at minimum number of samples: ', grid[indicesMAE[0][0]])
  

    

def RF_error(table, msl):
    """
    Now using the min_samples_leaf (msl) calculated above, 
    run the regressor 100 times using to calculate final importances and their errors
    """
        
    # read in different physics parameters we want to know the importances of
    M_s_list=tab['Mass_tot']
    sfr_list=tab['SFR_tot']
    metal_list=tab['12+log(O/H)_n2_r3']-8.69
    sfr_d_list=tab['SFR_D4000_inc_cut']
    random_list=np.random.random_sample(size=len(M_s_list))
    vd=np.log10(tab['SIGMA_BALMER']/100)
    
    # read in the variable we want to find importances relative to
    H_rat=tab['Balmer_dec']

    X=np.array([M_s_list, sfr_list, metal_list, sfr_d_list, vd, random_list]).T
    y=H_rat

    n_s=100 # repeat fit 100 times to get error
    
    ims=np.zeros([n_s, len(X[0])]) # initialise empty list to input the importances
    MAEtes=np.zeros(n_s)
    MSEtes=np.zeros(n_s)
    MAEtrs=np.zeros(n_s)
    MSEtrs=np.zeros(n_s)
    
    for i in range(n_s):
    
        # split sample into test and train, 50:50
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=i)
        
        # initialise regressor
        regr = RandomForestRegressor(random_state=i, min_samples_leaf=msl, n_jobs=-1)
        
        # fit regressor to training date
        regr.fit(X_train, y_train)
        
        # add importances to list
        ims[i]+=regr.feature_importances_
        
        # measure test train MAE for overffiting
        y_pred=regr.predict(X_test)
        MAEtes[i]+= mean_absolute_error(y_test, y_pred)
        MSEtes[i]+= mean_squared_error(y_test, y_pred)
        
        y_pred_tr=regr.predict(X_train)
        MAEtrs[i]+= mean_absolute_error(y_train, y_pred_tr)
        MSEtrs[i]+= mean_squared_error(y_train, y_pred_tr)
        
        print('Repetition ', i+1, ' out of ', n_s, ' done!')

    # calculate average importances, their error and variance
    im_av=np.mean(ims, axis=0)
    im_err=np.std(ims, axis=0)
    im_var=np.var(ims, axis=0)
    
    # calculate average and error of MAE and MSE for test and train samples
    MAEte_av=np.mean(MAEtes)
    MAEte_err=np.std(MAEtes)
    
    MSEte_av=np.mean(MSEtes)
    MSEte_err=np.std(MSEtes)
    
    MAEtr_av=np.mean(MAEtrs)
    MAEtr_err=np.std(MAEtrs)
    
    MSEtr_av=np.mean(MSEtrs)
    MSEtr_err=np.std(MSEtrs)
    
    
    print('Average importances are: ', im_av, '+/-', im_err)
    print('Average MAE on test sample is: %.3e +/- %.3e' %(MAEte_av,MAEte_err))
    print('Average MSE on test sample is: %.3e +/- %.3e' %(MSEte_av,MSEte_err))
    print('Average MAE on train sample is: %.3e +/- %.3e' %(MAEtr_av,MAEtr_err))
    print('Average MSE on train sample is: %.3e +/- %.3e' %(MSEtr_av,MSEtr_err))
    
    print('Fractional error from std is: ', im_err/im_av)
    print('Fractional error from var is: ', im_var/im_av)

    print('Median importances are: ', np.median(ims, axis=0))
    
    percen_err=np.percentile(ims,0.95, axis=0)-np.percentile(ims, 0.05, axis=0)
    print('95-5 percentile range is: ', percen_err)
    print('Variance of importances is: ', im_var)
    print('Fractional error from percentiles is: ', percen_err/im_av)
    
    # initiliase figure to show the importances
    fig2, ax2 = plt.subplots(figsize=(7,5))
        
    names=[r' M$_{\star}$', r' SFR$_{{\rm H}_\alpha}$', r'[O/H]',
           r'SFR$_{D4}$', r'$\sigma_{{\rm H}_{\alpha}}$',
           r'R']
    
    # combine importance averages, errors, and string names
    
    comb_dat=zip(im_av, im_err, names)
    
    # order from highest to lowest importance
    comb_dat=sorted(comb_dat, reverse=True)
    
    # unpack in this order
    im_av_sorted=[comb_dat[num][0] for num in range(len(comb_dat))]
    im_err_sorted=[comb_dat[num][1] for num in range(len(comb_dat))]
    names_sorted=[comb_dat[num][2] for num in range(len(comb_dat))]
    
    te_tr_text=str('MAE train, test =[%.2f, %.2f]' %(MAEtr_av, MAEte_av))+str('\n'+'MSE train, test =[%.2f, %.2f]' %(MSEtr_av, MSEte_av))

    h=ax2.bar(names_sorted, im_av_sorted, color = 'b', edgecolor = 'black', 
              yerr=im_err_sorted, alpha=.5, label=te_tr_text, zorder=3, capsize=10)
    
    ax2.set_ylabel('Relative Importance')
    ax2.set_ylim([0,.6])
    ax2.legend(loc='center right')
      
    
    fig_path=path+'/final_figs_paper/revised_plots/RF_imp_bar_chart.png'
    
    fig2.savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
    
    
    # initialise plot to check for overfitting visually
    
    # calculate 1:1 line
    x0=min([min(y_train), min(y_test)])
    x_s=np.linspace(x0, 6, len(y_train))
    y_s=x_s
    
    fig3, ax3 = plt.subplots(figsize=(7,5))
    ax3.grid(True)

    # make contours for test and train samples
    A,B, pdf, levels = make_contours(y_train, y_pred_tr)
    ax3.contour(A,B, pdf.T, levels=levels, colors='orange')#, label='Train')
    
    A,B, pdf, levels = make_contours(y_test, y_pred)
    ax3.contour(A,B, pdf.T, levels=levels, colors='blue')#, label='Test')
    
    ax3.legend(['Train', 'Test'])
    ax3.plot(x_s, y_s)
    ax3.set_xlabel(r'Input H$_{\alpha}$/H$_{\beta}$')
    ax3.set_ylabel(r'Predicted H$_{\alpha}$/H$_{\beta}$')
    ax3.set_xlim([2.9,6.7])
    ax3.set_ylim([3.4, 5.9])

    ax3.set_title(str_title)

    ax3.tick_params(which='both',bottom=True,top=True,right=True)
    ax3.xaxis.set_minor_locator(MultipleLocator(.1))
    ax3.yaxis.set_minor_locator(MultipleLocator(.1))
      
    fig_path=path+'/final_figs_paper/revised_plots/RF_tt_.png'
    
    fig3.savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
