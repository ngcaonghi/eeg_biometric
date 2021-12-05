import numpy as np
import mne
import glob, os

'''
Script for producing training and testing data.
'''

def lowpass_filter(arr, sfreq, lowpass):
    '''
    Performs low-pass filtering.
    '''
    return mne.filter.filter_data(arr, sfreq=sfreq, l_freq=None, h_freq=lowpass)

def add_epochs(X_lst, y_lst, X, y, sfreq, epoch_len, overlap):
    '''
    Add epochs to the list of data for training or testing.
    '''
    window = epoch_len * sfreq
    for i in np.arange(X.shape[1] // (window * (1-overlap))) * sfreq:
        start = int(i)
        stop = int(i + window)
        epoch = X[:, start:stop]
        # only use epochs with length == window
        if epoch.shape[-1] == window:
            X_lst.append(epoch)
            y_lst.append(y)

def train_test_split(sfreq=256, 
                    lowpass=50, 
                    epoch_len=10,
                    overlap=.9,
                    parent_folder=None,
                    data_folder=None):
    '''
    Lowpass filter, cut data into epochs, and split them into training and 
    test sets.

    NOTE: the third session of every subject is used as test data.

    Parameters:
        - sfreq (int): sampling rate. Default at 256.
        - lowpass (int, float): lowpass frequency. Default at 50.
        - epoch_len (int, float): the length of every epoch in seconds. Default
        at 10 seconds.
        - overlap (float): the proportion by which every pair of contiguous
        epochs overlaps. Must be within [0, 1]. Default at 0.9.
        - parent_folder (str): the parent directory of the folders containing 
        preprocessed files. Default is None under the assumption that this
        script file is already in the parent directory.
        - data_folder (str): the folder storing the training and test data.
        Default is None, which entails dumping those data to the same folder
        hosting this script.

    Return: none

    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    if parent_folder != None:
        subject_dirs = glob.glob(os.path.abspath(parent_folder)+os.sep+'*') 
    else:
        parent_folder = '.'+os.sep
        subject_dirs = glob.glob('.' + os.sep + '*')
    
    for d in subject_dirs:
        subject = d.split(os.sep)[-1]
        for session_dir in glob.glob(d+os.sep+'*.npy'):
            X = lowpass_filter(np.load(session_dir), sfreq, lowpass)
            session = session_dir.split(os.sep)[-1].rstrip('.npy')
            if session == subject + '_s1' or session == subject + '_s2':
                # NOTE: sessions 1 and 2 used for training, session 3 for test
                add_epochs(X_train, y_train, X, subject, sfreq, epoch_len, overlap)
            else:
                add_epochs(X_test, y_test, X, subject, sfreq, epoch_len, overlap)

    if data_folder == None:
        data_folder = '.' + os.sep
    else:
        data_folder = os.path.abspath(data_folder) + os.sep
    np.save(data_folder + 'X_train.npy', X_train)
    np.save(data_folder + 'X_test.npy', X_test)
    np.save(data_folder + 'y_train.npy', y_train)
    np.save(data_folder + 'y_test.npy', y_test)

if __name__ == '__main__':
    train_test_split()