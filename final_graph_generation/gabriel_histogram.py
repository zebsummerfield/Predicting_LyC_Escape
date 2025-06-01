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

os.chdir('/Users/') 
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


def pca_n(vari):
    dim=len(vari)
    if dim==3:

        # First we need the Spearmann rank correlation coefficients 
        
        
        r_AB=np.zeros((len(vari), len(vari)))
        indices=np.arange(0, len(vari), 1)
        
        for i in indices:
            for j in indices[(indices!=i)]:
                r_ij=ss.spearmanr(vari[i], vari[j])
                r_AB[i,j]=r_ij[0]
        
        # now to make the PCA matrix
                
        r_AB_C=np.zeros((len(vari), len(vari)))
        
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    r_ij_k=(r_AB[i,j]-r_AB[i,k]*r_AB[j,k])/(
                        np.sqrt(1-r_AB[i,k]**2)*np.sqrt(1-r_AB[j,k]**2))
                    r_AB_C[i,j]=r_ij_k 
                    
        tantheta=r_AB_C[0,2]/r_AB_C[1,2]
        return r_AB_C, tantheta
    
    elif dim==4:
        
        # try with 4 variables 
                
        r_AB=np.zeros((len(vari), len(vari)))
        
        indices=np.arange(0, len(vari), 1)
        
        for i in indices:
            for j in indices[(indices!=i)]:
                r_ij=ss.spearmanr(vari[i], vari[j])
                r_AB[i,j]=r_ij[0]
               
        
        r_AB_C=np.zeros((len(vari), len(vari), len(vari)))
        
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    r_ij_k=(r_AB[i,j]-r_AB[i,k]*r_AB[j,k])/(
                        np.sqrt(1-r_AB[i,k]**2)*np.sqrt(1-r_AB[j,k]**2))
                    r_AB_C[i,j,k]=r_ij_k
            
        r_AB_CD=np.zeros((len(vari), len(vari)))
             
        for i in indices:
            for j in indices[(indices!=i)]:
                for k in indices[(indices!=i) & (indices!=j)]:
                    l=indices[(indices!=i) & (indices!=j) & (indices!=k)]
        
                    r_ij_kl=(r_AB_C[i,j,k]-r_AB_C[i,l,k]*r_AB_C[j,l,k])/(
                                np.sqrt(1-r_AB_C[i,l,k]**2)*np.sqrt(1-r_AB_C[j,l,k]**2))
                    
                    r_AB_CD[i,j]=r_ij_kl
            tantheta=0
        return r_AB_CD, tantheta 
    
def pcc_err(vari_pcc, theta=False):

    n_s=100 
    # repeat n_s times
    
    s_s=len(vari_pcc[0]) 
    # draw s_s elements with replacement from sample each time i.e. its put back so can be picked again
    
    random.seed(10)
    dat=vari_pcc.T
    pcc_sam=[[]]
    theta_sam=[]
    
    for i in range(n_s):
        dat_sam=resample(dat, n_samples=s_s, random_state=i)  
        r_AB_C, tantheta=pca_n(dat_sam.T)
        pcc_sam.append(r_AB_C.ravel())
        theta_sam.append(np.arctan(tantheta))
    
    del pcc_sam[0]
    
    pcc_std_l=np.std(pcc_sam, axis=0)
    
    dim=len(vari_pcc)
    pcc_std=pcc_std_l.reshape(dim,dim)
    r_AB_C, tantheta=pca_n(vari_pcc)
    
    if theta:
        theta_err=np.std(theta_sam)
        return np.arctan(tantheta), theta_err
    # returns theta of normal PCC and then error from bootstrapping method!
    
    else:
        print(r_AB_C)
        print(pcc_std)
        return r_AB_C, pcc_std

fig, axs = plt.subplots(figsize=(7,5))

h = axs.scatter(x_vals, y_vals, c = cval, s=30)
axs.errorbar(x_vals, y_vals, fmt= 'none',
                xerr = x_err, yerr = y_err,
                alpha=0.1)

axs.set_box_aspect(1) # make physical size of each axis same size



start_t=[.5,.5]
width=0.02
r=.2
# d_b=.05

(a_t, b_t)=start_t
dat=h.get_offsets()
x,y=dat.T
z=h.get_array()
d_b=0
vari_pcc=np.array([x,y,z])

theta, theta_err=pcc_err(vari_pcc, theta=True)
if theta<0:
    (da,db)=(r*np.sin(-theta), -r*np.cos(-theta)) # for active
    txt=str(r'$\theta$=' + str(round(theta*180/np.pi+180, 2))+ r'$\pm$' + str(round(theta_err*180/np.pi, 2)) + r'$^{\circ}$')

elif theta>0:
    (da,db)=(r*np.sin(theta), r*np.cos(theta)) # for passive also
    txt=str(r'$\theta$=' + str(round(theta*180/np.pi, 2))+ r'$\pm$' + str(round(theta_err*180/np.pi, 2)) + r'$^{\circ}$')

axs.set_title(txt)

axs.arrow(a_t, b_t, da, db, width=width, ec='black', fc='grey', transform=axs.transAxes, zorder=5)
# axs.transAxes forces the arrow to be plot on a 1:1 scale so that the angle it shows is understandable