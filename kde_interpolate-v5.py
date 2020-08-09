#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from astropy.io import fits
import glob


# In[2]:


def KDE_pdf(data, kernel = 'gaussian'):
    #Reshape
    data = data.reshape(-1,1)
    
    #Define Bandwidth as the pratical computation
    bandwidth = np.abs(1.06*np.std(data)*len(data)**(-1./5.))
    
    #Fit Kernel Density Estimation from dataset
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    
    return kde

#Estimate KDE Log Probability for a given dataset 
def proba_kde(data_points, kde):
    log_dens = kde.score_samples(data_points)
    return log_dens


def log_like(dataset, wtheta_values):
    #Create Kernels and compute means
    aux_mean = wtheta_values
    aux_kde = []
    for i in dataset.columns:
        aux_kde.append(KDE_pdf(dataset[i].values, kernel = 'gaussian'))
    
    #Choosen DataPoints to estimate Likelihood 
    aux_mean = np.array(aux_mean).reshape(-1,1) #We need to substitue for the theoretical value

    #Estimate Log Likelihood 
    log_like = 0
    for i in range(len(aux_mean)):
        log_like = log_like + proba_kde(aux_mean[i].reshape(-1,1), kde = aux_kde[i])

    return log_like[0]


# In[3]:


input_fits = fits.open('/Users/iltomo/cosmosis-docker/cosmosis/cosmosis-standard-library/likelihood/des-y1/2pt_NG_mcal_1110.fits')
theta_des = input_fits['wtheta'].data['ANG']
wtheta_des = input_fits['wtheta'].data['VALUE']

###

#filenames = glob.glob("multi_galaxy_xi/bin*.txt")

mg_xi = {}

for i in range(1,6):
    filenames = glob.glob("multi_galaxy_xi/bin_%d_%d_*.txt"%(i,i))
    for j in filenames:
    
        col = '%s'%j
        col = col.replace('multi_galaxy_xi/','')
        col = col.replace('.txt','')

        aux = np.genfromtxt(j)
    
        mg_xi[col] = aux#[aux.columns[0]]

mg_xi = pd.DataFrame(mg_xi)
    
theta = pd.read_csv("multi_galaxy_xi/theta.txt")
mg_xi['theta'] = pd.to_numeric(theta.iloc[1:,0].values)

###

for i in range(1,6):
    wtheta_new = np.zeros ([100,20])
    for n in range (1,101):
        name = 'bin_%d_%d_'%(i,i) + str (n)
        mg_xi_interpolate = interp1d (mg_xi ['theta'], mg_xi[name])
        wtheta_new[n-1] = mg_xi_interpolate (theta_des[(i-1)*20:i*20])
    
    wtheta_trn = pd.DataFrame(wtheta_new.T)
    
    aux = ['col_'+ str(x) for x in range(np.shape(wtheta_trn)[1])]
    wtheta_trn.columns = aux

    print('bin_%d_%d_:'%(i,i), log_like(wtheta_trn,input_fits['wtheta'].data['VALUE'][(i-1)*20:i*20]),'\n')



