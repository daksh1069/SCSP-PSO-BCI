#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all the Necessary Libraries
import numpy as np
import pandas as pd
import scipy.io
from sklearn import svm, pipeline, base, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import functools
import os.path, zipfile
import matplotlib.pyplot as plt
import warnings
from scipy import signal


# In[2]:


# Loading Dataset

# Read DataSet BCI Competition III, DataSet IVa - Training
sub1_100hz_training = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\100 Hz\MATLAB\sub1\100Hz\data_set_IVa_aa.mat", struct_as_record=True)
sub2_100hz_training = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\100 Hz\MATLAB\sub2\100Hz\data_set_IVa_al.mat", struct_as_record=True)
sub3_100hz_training = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\100 Hz\MATLAB\sub3\100Hz\data_set_IVa_av.mat", struct_as_record=True)
sub4_100hz_training = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\100 Hz\MATLAB\sub4\100Hz\data_set_IVa_aw.mat", struct_as_record=True)
sub5_100hz_training = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\100 Hz\MATLAB\sub5\100Hz\data_set_IVa_ay.mat", struct_as_record=True)


# Read DataSet BCI Competition III, DataSet IVa - True Label
sub1_100hz_true_label = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\True Labels\true_labels_aa.mat", struct_as_record=True)
sub2_100hz_true_label = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\True Labels\true_labels_al.mat", struct_as_record=True)
sub3_100hz_true_label = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\True Labels\true_labels_av.mat", struct_as_record=True)
sub4_100hz_true_label = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\True Labels\true_labels_aw.mat", struct_as_record=True)
sub5_100hz_true_label = scipy.io.loadmat(r"C:\Users\Daksh kumar\BCI CodeBase\Relevant DataSet\Competition III\4A\True Labels\true_labels_ay.mat", struct_as_record=True)


# In[4]:


# Loading Important Global Data
sample_rate = 100
# The time window (in samples) to extract for each trial, here 0.5 -- 3.5 seconds
win = np.arange(int(0.5*sample_rate), int(3.5*sample_rate))
nsamples = len(win)

# SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
# extra dimensions in the arrays. This makes the code a bit more cluttered


m = sub1_100hz_training



sample_rate = m['nfo']['fs'][0][0][0][0]
#EEG = m['cnt'].T
#nchannels_yt, nsamples_yt = EEG.shape

#channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]
#labels = np.zeros((1, nsamples), int)
#labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in m['mrk']['className'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]
nclasses = len(cl_lab)
nevents = len(event_onsets)


# # Loading Previously Written Modules that are to be Reused - 

# In[5]:


# Calculate the log(var) of the trials

def logvar(trials):
    '''
    Calculate the log-var of each channel.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal.
        
    Returns
    -------
    logvar - 2d-array (channels x trials)
        For each channel the logvar of the signal
    '''
    return np.log(np.var(trials, axis=1))


# In[6]:



# Below is a function to visualize the logvar of each channel as a bar chart:

def plot_logvar(trials,shapevar):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12,5))
    nchannels, _ = shapevar.shape
    print(nchannels)
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(cl_lab)


# In[7]:


# PSD Functions

from matplotlib import mlab

def psd(trials):
    '''
    Calculates for each trial the Power Spectral Density (PSD).
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    
    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.  
    freqs : list of floats
        Yhe frequencies for which the PSD was computed (useful for plotting later)
    '''
    
    ntrials = trials.shape[2]
    nchannels = trials.shape
    trials_PSD = np.zeros((nchannels, 151, ntrials))

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()
                
    return trials_PSD, freqs



# In[8]:


import matplotlib.pyplot as plt

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().
    
    Parameters
    ----------
    trials : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd() 
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))
    
    nchans = len(chan_ind)
    
    # Maximum of 3 plots per row
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)
    
    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows,ncols,i+1)
    
        # Plot the PSD for each class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
    
        # All plot decoration below...
        
        plt.xlim(1,30)
        
        if maxy != None:
            plt.ylim(0,maxy)
    
        plt.grid()
    
        plt.xlabel('Frequency (Hz)')
        
        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()
        
    plt.tight_layout()

    


# In[9]:


# Extracting Trials

def ExtractTrial(X,event_onsets,sample_rate,cl_lab,event_codes):
    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 3.5 seconds
    win = np.arange(int(0.5*sample_rate), int(3.5*sample_rate))
    #print(event_onsets.shape)
    nchannels, nsamples = np.array(X).shape
    #print(nchannels,nsamples)
    #print(cl_lab)
    # Length of the time window
    nsamples = len(win)
    #print(nsamples)
    #print(np.unique(event_codes))
    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        #print(cl,code)
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        #print(cl_onsets)
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
    
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = np.array(X)[:, win+onset]
    print(trials[cl1].shape,trials[cl2].shape)
    return trials


# In[10]:


# BPF Function 

import scipy.signal 

def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])
    nchannels,nsamples,ntrials = trials.shape
    # Applying the filter to each trial
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
    
    return trials_filt


# In[11]:


# CSP Function

from numpy import linalg

def cov(trials):
    ''' Calculate the covariance for each trial and return their average '''
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)

def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = linalg.svd(sigma)
    return U.dot( np.diag(l ** -0.5) )

def csp(trials_r, trials_f):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right hand movement trials
        trials_f - Array (channels x samples x trials) containing foot movement trials
    returns:
        Mixing matrix W
    '''
    cov_r = cov(trials_r)
    cov_f = cov(trials_f)
    P = whitening(cov_r + cov_f)
    B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
    W = P.dot(B)
    return W

def apply_mix(W, trials,shapevar):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    ntrials = trials.shape[2]
    nchannels, _ = shapevar.shape
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    print(trials_csp.shape)
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp




# Plot Scatter

def plot_scatter(left, foot):
    plt.figure()
    plt.scatter(left[0,:], left[-1,:], color='b')
    plt.scatter(foot[0,:], foot[-1,:], color='r')
    plt.xlabel('Last component')
    plt.ylabel('First component')
    plt.legend(cl_lab)


# In[15]:


# Function for Calculating Channel Variance - Also Used for Dimenionsanality Reduction

class ChanVar(base.BaseEstimator, base.TransformerMixin):
    def fit( X, y):return X,y
    def transform(X):
        return np.var(X, axis=1)  # X.shape = (trials, channels, time)
    


# In[16]:


# SCSP Function


from numpy import linalg

def scov(trials,k):
    ''' Calculate the covariance for each trial and return their average '''
    ntrials = trials.shape[2]
    print(ntrials)
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    # Creating Chunks
    chunk = trials[:,:,:k]
    nchunks = chunk.shape[2]
    #print(chunk.shape)
    chunk_covs = []
    chunk_covs = [ chunk[:,:,i].dot(chunk[:,:,i].T) / nsamples for i in range(nchunks) ]
    
    return np.mean(covs, axis=0),np.mean(chunk_covs, axis=0)

def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = linalg.svd(sigma)
    return U.dot( np.diag(l ** -0.5) )

def scsp(trials_r, trials_f,mu = 0.5,k = 3):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right hand movement trials
        trials_f - Array (channels x samples x trials) containing foot movement trials
    returns:
        Mixing matrix W
    '''
    cov_r,chunk_r = scov(trials_r,k)
    cov_f,chunk_f = scov(trials_f,k)
    del_r = abs(chunk_r-cov_r)
    del_f = abs(chunk_f-cov_f)
    del_r = del_r/k
    del_f = del_f/k
    #print("Print chunk_r and chunk_f shape")
    #print(chunk_r.shape,chunk_f.shape)
    #print("Print Cov_r and Cov_f shape")
    #print(cov_r.shape,cov_f.shape)
    P = whitening(cov_r + cov_f + mu*(del_r + del_f) )
    B, _, _ = linalg.svd( P.T.dot(cov_f).dot(P) )
    W = P.dot(B)
    return W

def apply_mix(W, trials,shapevar):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    ntrials = trials.shape[2]
    nchannels, _ ,__ = shapevar.shape
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    #print(trials_csp.shape)
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp


# In[17]:



# Reading Data
def load_mat(mat_train, mat_test, rec_id):
    mat = mat_train
    mat_true = mat_test
    '''Load BCI Comp. 3.4a specific Matlab files.'''
    #mat = io.loadmat(mat_train, struct_as_record=True)
    #mat_true = io.loadmat(mat_test, struct_as_record=True)

    # get simple info from MATLAB files
    X, nfo, mrk = .1 * mat['cnt'].astype(float).T, mat['nfo'], mat['mrk']
    X, nfo, mrk = X.astype(np.float32), nfo[0][0], mrk[0][0]
    sample_rate = float((nfo['fs'])[0][0])
    dt = np.ones(X.shape[1]-1) / sample_rate
    chan_lab = [str(c[0]) for c in nfo['clab'].flatten()]

    # extract labels from both MATLAB files
    offy = mrk['pos'].flatten()
    tr_y = mrk['y'].flatten()
    all_y = mat_true['true_y'].flatten()
    assert np.all((tr_y == all_y)[np.isfinite(tr_y)]), 'labels do not match.'

    class_lab = [str(c[0]) for c in (mrk['className'])[0]]
    events = np.vstack([all_y, offy, offy + 3.5 * sample_rate]).astype(int)
    event_lab = dict(zip(np.unique(events[0]), class_lab))

    folds = np.where(np.isfinite(tr_y), -1, 1).tolist()
    
    #print('Format For Printing - X,dt,chan_lab,events,event_lab,folds,rec_id')
    #print(X)
    #print(dt)
    #print(chan_lab)
    #print(events)
    #print(event_lab)
    #print(folds)
    #print(rec_id)
    #print(X[0])
    return regions(X,dt,chan_lab,events,event_lab,folds,rec_id)
    


# In[19]:


def regions(X,dt,chan_lab,events,event_lab,folds,rec_id):
    
    Regions = dict()
    Region_X = []
    Region_chan_lab = []
    Region_events1 = []
    Region_events2 = []
    Region_events3 = []
    Region_event_lab = []
    Region_folds = []
    
    for i in range(0,len(chan_lab)):
        Region_X.append(X[i])
        Region_chan_lab.append(chan_lab[i])
        Region_event_lab.append(event_lab)
    Region_events1 = events[0]
    Region_events2 = events[1]
    Region_events3 = events[2]
    Region_folds = folds
    #print(Region_X,Region_chan_lab,Region_events1,Region_events2,Region_events3,event_lab,Region_folds)
    Regions = {'X' :Region_X ,'chan_lab' :Region_chan_lab ,'events1' : Region_events1,'events2' : Region_events2,'events3' : Region_events3,'event_lab' :Region_event_lab ,'folds' :Region_folds }
    return Regions
    


# In[20]:
# SVM Functions
def SVMModel(X,y,test_size):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    model = svm.SVC(kernel='rbf',gamma = 1)
    model.fit(X_train,y_train)
    return model,X_test,y_test

def SVMAccuracy(model,X_test,y_test):
    y_pred = model.predict(X_test)
    #print (metrics.classification_report(y_test, y_pred))
    #print(accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)




# In[21]:
# Data Creation and Combination Module
def CombineData(SCSP1,SCSP2):
    SCSP = np.dstack((SCSP1,SCSP2),)
    #print(SCSP.shape)
    return SCSP
def CreateY(trials_filt):
    y_1 = np.zeros(trials_filt[cl1].shape[2])
    y_1.fill(1)
    y_2 = np.zeros(trials_filt[cl2].shape[2])
    y_2.fill(0)
    y = np.hstack((y_1,y_2))
    #train_region1_combT = np.transpose(train_region1_comb,(2,1,0))
    #print(y.shape)
    return y



# In[22]:
# Extract the actual Population - Not Gonna Use though
def ActualData():
    for i in range(0,options['N']):
        temp1 = []
        temp2 = []
        for j in range(0,len(Population[i])):
            if(Population[i][j] == 1):
                temp1.append(trials_filt[cl1][j]),
                temp2.append(trials_filt[cl2][j])
        #print(len(temp1),len(temp2))
        len1 = len(temp1)
        temp1 = np.asarray(temp1)
        temp2 = np.asarray(temp2)
        #print(temp1.shape,temp2.shape)
        #print(temp[0][0].shape)
        print(len1)
        pso_filt = dict()
        for j in range(0,len1):
            pso_filt[j] = {
                cl1 : temp1,
                cl2 : temp2
            }
            print(pso_filt[i][cl1].shape,i)
    return pso_filt




# In[23]:

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



# In[24]:





# In[25]:




