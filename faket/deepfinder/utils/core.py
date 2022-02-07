# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

# This file contains classes/functions that are judged not necessary for the user.

import numpy as np
import h5py
import os
import sys

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

from . import common as cm

class DeepFinder:
    def __init__(self):
        self.obs_list = [observer_print]

    # Useful for sending prints to GUI
    def set_observer(self, obs):
        self.obs_list.append(obs)

    # "Master print" calls all observers for prints
    def display(self, message):
        for obs in self.obs_list: obs.display(message)

    # For checking inputs:
    def is_3D_nparray(self, v, varname):
        if type(v)!=np.ndarray:
            self.display('DeepFinder message: variable "'+varname+'" is '+str(type(v))+'. Expected is numpy array.')
            sys.exit()
        if len(v.shape)!=3:
            self.display('DeepFinder message: variable "'+varname+'" is a '+str(len(v.shape))+'D array. Expected is a 3D array.')
            sys.exit()

    def is_int(self, v, varname):
        if type(v)!=int and type(v)!=np.int8 and type(v)!=np.int16:
            self.display('DeepFinder message: variable "' + varname + '" is ' + str(type(v)) + '. Expected is int.')
            sys.exit()

    def is_positive_int(self, v, varname):
        self.is_int(v, varname)
        if v<=0:
            self.display('DeepFinder message: variable "'+varname+'" is <=0. Expected is >0.')
            sys.exit()

    def is_multiple_4_int(self, v, varname):
        self.is_int(v, varname)
        if v % 4 != 0:
            self.display('DeepFinder message: variable "' + varname + '" should be a multiple of 4.')
            sys.exit()

    def is_str(self, v, varname):
        if type(v) != str:
            self.display('DeepFinder message: variable "' + varname + '" is ' + str(type(v)) + '. Expected is str.')
            sys.exit()

    def is_h5_path(self, v, varname):
        self.is_str(v, varname)
        s = os.path.splitext(v)
        if s[1] != '.h5':
            self.display('DeepFinder message: "'+str(varname)+'" points to '+s[1]+', expected is .h5')
            sys.exit()

    def is_list(self, v, varname):
        if type(v)!=list:
            self.display('DeepFinder message: variable "' + varname + '" is ' + str(type(v)) + '. Expected is list.')
            sys.exit()

    def are_lists_same_length(self, v, varname):
        if len(v[0])!=len(v[1]):
            self.display('DeepFinder message: the lists "' + varname[0] + '" and "' + varname[1] + '" need to be of same length.')
            sys.exit()

    def check_array_minsize(self, v, varname):
        lmin = v[1] # is expected to be int (e.g. patch length)
        if v[0].shape[0] < lmin and v[0].shape[1] < lmin and v[0].shape[2] < lmin:
            self.display('DeepFinder message: the array "' + varname[0] + '" has shape '+str(v[0].shape)+'. Needs to be larger than array "'+varname[1]+'", which has shape ('+str(v[1])+','+str(v[1])+','+str(v[1])+').')
            sys.exit()

# Following observer classes are needed to send prints to GUI:
class observer_print:
    def display(message):
        print(message)

class observer_gui:
    def __init__(self, pyqt_signal):
        self.sig = pyqt_signal
    def display(self, message):
        self.sig.emit(message)

# Retrieves variable name as a str:
# Found here: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def retrieve_var_name(x, Vars=vars()):
    for k in Vars:
        if type(x) == type(Vars[k]):
            if x is Vars[k]:
                return k
    return None


# This functions loads the training set at specified paths.
# INPUTS:
#   path_data  : list of strings '/path/to/tomogram.ext'
#   path_target: list of strings '/path/to/target.ext'
#                The idx of above lists correspond to each other so that (path_data[idx], path_target[idx]) corresponds
#                to a (tomog, target) pair
#   dset_name  : can be usefull if files are stored as .h5
# OUTPUTS:
#   data_list  : list of 3D numpy arrays (tomograms)
#   target_list: list of 3D numpy arrays (annotated tomograms)
#                In the same way as for the inputs, (data_list[idx],target_list[idx]) corresponds to a (tomo,target) pair
# TODO: move to common?
def load_dataset(path_data, path_target, dset_name='dataset'):
    data_list   = []
    target_list = []
    for idx in range(0,len(path_data)):
        data   = cm.read_array(path_data[idx]  , dset_name)
        target = cm.read_array(path_target[idx], dset_name)

        if data.shape!=target.shape:
            print('DeepFinder message: tomogram and target are not of same size!')
            sys.exit()

        data_list.append(data)
        target_list.append(target)
    return data_list, target_list

# This function applies bootstrap (i.e. re-sampling) in case of unbalanced classes.
# Given an objlist containing objects from various classes, this function outputs an equal amount of objects for each
# class, each objects being uniformely sampled inside its class set.
# INPUTS:
#   objlist: list of dictionaries
#   Nbs    : number of objects to sample from each class
# OUTPUT:
#   bs_idx : list of indexes corresponding to the bootstraped objects
def get_bootstrap_idx(objlist,Nbs):
    # Get a list containing the object class labels (from objlist):
    Nobj = len(objlist)
    label_list = []
    for idx in range(0, Nobj):
        label_list.append(objlist[idx]['label'])

    lblTAB = np.unique(label_list)  # vector containing unique class labels

    # Bootstrap data so that we have equal frequencies (1/Nbs) for all classes:
    # ->from label_list, sample Nbs objects from each class
    bs_idx = []
    for l in lblTAB:
        bs_idx.append(np.random.choice(np.array(np.nonzero(np.array(label_list) == l))[0], Nbs))

    bs_idx = np.concatenate(bs_idx)
    return bs_idx

# Takes position specified in 'obj', applies random shift to it, and then checks if the patch around this position is
# out of the tomogram boundaries. If so, the position is shifted to that patch is inside the tomo boundaries.
# INPUTS:
#   tomodim: tuple (dimX,dimY,dimZ) containing size of tomogram
#   p_in   : int lenght of patch in voxels
#   obj    : dictionary obtained when calling objlist[idx]
#   Lrnd   : int random shift in voxels applied to position
# OUTPUTS:
#   x,y,z  : int,int,int coordinates for sampling patch safely
def get_patch_position(tomodim, p_in, obj, Lrnd):
    # sample at coordinates specified in obj=objlist[idx]
    x = int( obj['x'] )
    y = int( obj['y'] )
    z = int( obj['z'] )
        
    # Add random shift to coordinates:
    x = x + np.random.choice(range(-Lrnd,Lrnd+1))
    y = y + np.random.choice(range(-Lrnd,Lrnd+1))
    z = z + np.random.choice(range(-Lrnd,Lrnd+1))
    
    # Shift position if too close to border:
    if (x<p_in) : x = p_in
    if (y<p_in) : y = p_in
    if (z<p_in) : z = p_in
    if (x>tomodim[2]-p_in): x = tomodim[2]-p_in
    if (y>tomodim[1]-p_in): y = tomodim[1]-p_in
    if (z>tomodim[0]-p_in): z = tomodim[0]-p_in

    #else: # sample random position in tomogram
    #    x = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    y = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    #    z = np.int32( np.random.choice(range(p_in,tomodim[0]-p_in)) )
    
    return x,y,z

# Saves training history as .h5 file.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history.h5'
def save_history(history, filename):
    if os.path.isfile(filename): # if file exists, delete before writing the updated version
        os.remove(filename)      # quick fix for OSError: Can't write data (no appropriate function for conversion path)

    h5file = h5py.File(filename, 'w')

    # train and val loss & accuracy:
    dset    = h5file.create_dataset('acc', np.array(history['acc']).shape, dtype='float16')
    dset[:] = np.array(history['acc'], dtype='float16')
    dset    = h5file.create_dataset('loss', np.array(history['loss']).shape, dtype='float16')
    dset[:] = np.array(history['loss'], dtype='float16')
    dset    = h5file.create_dataset('val_acc', np.array(history['val_acc']).shape, dtype='float16')
    dset[:] = np.array(history['val_acc'], dtype='float16')
    dset    = h5file.create_dataset('val_loss', np.array(history['val_loss']).shape, dtype='float16')
    dset[:] = np.array(history['val_loss'], dtype='float16')

    # val precision, recall, F1:
    dset    = h5file.create_dataset('val_f1', np.array(history['val_f1']).shape, dtype='float16')
    dset[:] = np.array(history['val_f1'], dtype='float16')
    dset    = h5file.create_dataset('val_precision', np.array(history['val_precision']).shape, dtype='float16')
    dset[:] = np.array(history['val_precision'], dtype='float16')
    dset    = h5file.create_dataset('val_recall', np.array(history['val_recall']).shape, dtype='float16')
    dset[:] = np.array(history['val_recall'], dtype='float16')

    h5file.close()
    return

def read_history(filename):
    history = {'acc': None, 'loss': None, 'val_acc': None, 'val_loss': None, 'val_f1': None, 'val_precision': None, 'val_recall': None}

    h5file = h5py.File(filename, 'r')
    # train and val loss & accuracy:
    history['acc']      = h5file['acc'][:]
    history['loss']     = h5file['loss'][:]
    history['val_acc']  = h5file['val_acc'][:]
    history['val_loss'] = h5file['val_loss'][:]
    # val precision, recall, F1:
    history['val_f1']        = h5file['val_f1'][:]
    history['val_precision'] = h5file['val_precision'][:]
    history['val_recall']    = h5file['val_recall'][:]

    h5file.close()
    return history

# Plots the training history as several graphs and saves them in an image file.
# Validation score is averaged over all batches tested in validation step (steps_per_valid)
# Training score is averaged over last N=steps_per_valid batches of each epoch.
#   -> This is to have similar curve smoothness to validation.
# INPUTS:
#   history: dictionary object containing lists. These lists contain scores and metrics wrt epochs.
#   filename: string '/path/to/net_train_history_plot.png'
def plot_history(history, filename):
    Ncl = len(history['val_f1'][0])
    legend_names = []
    for lbl in range(0,Ncl):
        legend_names.append('class '+str(lbl))

    epochs = len(history['val_loss'])
    steps_per_valid = len(history['val_loss'][0])

    hist_loss_train = []
    hist_acc_train = []
    hist_loss_valid = []
    hist_acc_valid = []
    hist_f1 = []
    hist_recall = []
    hist_precision = []
    for e in range(epochs):
        hist_loss_train.append(np.mean(history['loss'         ][e][-steps_per_valid:]))
        hist_acc_train.append( np.mean(history['acc'          ][e][-steps_per_valid:]))

        hist_loss_valid.append(np.mean(history['val_loss'     ][e]))
        hist_acc_valid.append( np.mean(history['val_acc'      ][e]))

        # array_f1 = np.array(history['val_f1'       ][e]) # easier to achieve desired averaging (per class) with np array
        # array_re = np.array(history['val_recall'   ][e])
        # array_pr = np.array(history['val_precision'][e])
        hist_f1.append(        np.mean(np.array(history['val_f1'       ][e]), axis=0))
        hist_recall.append(    np.mean(np.array(history['val_recall'   ][e]), axis=0))
        hist_precision.append( np.mean(np.array(history['val_precision'][e]), axis=0))

    fig = plt.figure(figsize=(15,12))
    plt.subplot(321)
    plt.plot(hist_loss_train, label='train')
    plt.plot(hist_loss_valid, label='valid')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(323)
    plt.plot(hist_acc_train, label='train')
    plt.plot(hist_acc_valid, label='valid')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(hist_f1)
    plt.ylabel('F1-score')
    plt.xlabel('epochs')
    plt.legend(legend_names)
    plt.grid()

    plt.subplot(324)
    plt.plot(hist_precision)
    plt.ylabel('precision')
    plt.xlabel('epochs')
    plt.grid()

    plt.subplot(326)
    plt.plot(hist_recall)
    plt.ylabel('recall')
    plt.xlabel('epochs')
    plt.grid()

    fig.savefig(filename)
