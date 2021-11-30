import pyprep
import mne
import numpy as np 
import scipy as sp 
import glob, os

def _loadmat(filename):
    '''
    ============================================================================

    Code contributed by jpapon on Stack Overflow:
    https://stackoverflow.com/a/29126361

    ============================================================================

    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

class Epoch():
    '''
    Class for extracting the desired epoch out of raw data and performing PREP 
    preprocessing on it.

    Attributes:
        - mat (dict): mat binary converted to dictionary of arrays and 
        mat structs.
        - mat_dir (str): directory to .mat binary file.
        - event (str): name of event of interest. Default at 'CLOSED EYES'.
        - name (str): name of session.
        - events (np.array): array of all events .
        - record (np.array): a (time_bins, channels) array of EEG recording
        - ch_names (list): list of channel names. Default is ['F3', 'FC5', 
        'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 
        'F4'].
        - ch_types (str): type of channels as a parameter for MNE create_info
        method.
        - sfreq (int, float): sample frequency. Default at 256.
        - montage_kind (str): name of montage kind for MNE create_info method.
        Default is 'standard_1020'.
        - bad (str): most severe type of bad channels.
        - prep (pyprep.PrepPipeline): the PrepPipeline object created by pyprep.
        - epoch (np.array): a a (time_bins, channels) array of the epoch of
        interest.
        - mask (2-tuple): a tuple indicating the start and stop indices of the
        self.record array by which it is cropped to produce an epoch.

    Methods:
        - __init__(self, mat_dir, event, ch_names, ch_types, sfreq, montage_kind)
        - _find_mask(self)
        - _crop(self)
        - _fit_whole(self)
        - fit(self, ch_types, ref_chs, reref_chs, line_freq, max_prep_iter, 
        filter_kwargs) 
        - report(self)
        - save_numpy(self, filename)
    '''
    def __init__(self, mat_dir, 
                event='CLOSED EYES', 
                ch_names=['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 
                          'T8', 'F8', 'AF4', 'FC6', 'F4'],
                ch_types='eeg',
                sfreq=256,
                montage_kind='standard_1020'):
        '''
        Init method.

        Parameters: 
            - mat_dir (str): directory to .mat binary file.
            - event (str): name of event of interest. Default at 'CLOSED EYES'.
            - ch_names (list): list of channel names. Default is ['F3', 'FC5', 
            'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 
            'F4'].
            - ch_types (str): type of channels as a parameter for MNE 
            create_info method.
            - sfreq (int, float): sample frequency. Default at 256.
            - montage_kind (str): name of montage kind for MNE create_info 
            method. Default is 'standard_1020'.
        '''
        self.mat = _loadmat(mat_dir)
        self.mat_dir = mat_dir
        self.event = event
        self.name = os.path.splitext(os.path.basename(mat_dir))[0]
        self.events = self.mat['events']
        self.record = self.mat['recording']
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.sfreq = sfreq
        self.montage_kind = montage_kind
        self.bad = None
        self.prep = None
        self.epoch = None
        self.mask = None

    def _find_mask(self):
        '''
        Find start and stop indices to crop raw array.
        '''
        event_split = self.event.upper().split()
        event_id = None
        for i, event_struct in enumerate(self.events):
            desc = event_struct[-1].__dict__.values()
            if sum([x in desc for x in event_split]) == len(event_split):
                event_id = i
        trigger_start = self.events[event_id][0]
        trigger_stop = self.events[event_id][1]
        start_id = 0
        stop_id = 0
        for id, t in enumerate(self.record.T[-1]):
            if t >= trigger_start:
              start_id = id
              break
        for id, t in enumerate(self.record.T[-1]):
            if t >= trigger_stop:
              stop_id = id
              break  
        self.mask = (start_id, stop_id)
        return self.mask
    
    def _crop(self):
        '''
        Crop raw array to yield the epoch of interest.
        '''
        if self.event != None:
            start_id, stop_id = self._find_mask()
            self.epoch = self.record[start_id : stop_id]
        else:
            self.epoch = self.record

    def _fit_whole(self, info, montage, prep_params, filter_kwargs):
        '''
        If the epoch is too bad, switch to running the pipeline again on 
        the whole raw array.
        '''
        np_eeg = self.record[:, 2:-1].T
        mne_eeg = mne.io.RawArray(np_eeg, info)
        try:
            prep = pyprep.PrepPipeline(mne_eeg, prep_params, montage, 
                                       filter_kwargs=filter_kwargs,
                                       ransac=True)
            prep.fit()
        except(OSError, ValueError):
            self.bad = 'ransac_whole'
            prep = pyprep.PrepPipeline(mne_eeg, prep_params, montage, 
                                       filter_kwargs=filter_kwargs,
                                       ransac=False)
            try:
                prep.fit()
                self.prep = prep
                self.prep.EEG_clean = \
                                  self.prep.EEG_clean[self.mask[0]:self.mask[1]]
            except(ValueError):
                self.bad = 'robust_whole'
                raise ValueError('Data is too bad to perform PREP')
        self.prep = prep
        self.prep.EEG_clean = self.prep.EEG_clean[self.mask[0] : self.mask[1]]

    def fit(self, 
            ch_types='eeg',
            ref_chs='eeg',
            reref_chs='eeg',
            line_freq=60,
            max_prep_iter=4,
            filter_kwargs={}):
        '''
        Perform PREP on the epoch.

        Parameters:
            - ch_types (str): channel type. Default at 'eeg'.
            - ref_chs (list, str): reference channels set up during 
            experimentation. The input can be a list of strings representing the
            names of the selected channels or 'eeg' if all channels were 
            considered. Default at 'eeg'. 
            - reref_chs (list, str): re-referencing channels. The input can be a 
            list of strings representing the names of the selected channels or 
            'eeg' if all channels are to be considered. Default at 'eeg'.
            - line_freq (int): line noise frequency. Default at 60.
            - max_prep_iter (int): maximum number of iterations for the robust
            referencing routine. Default at 4.
            - filter_kwargs (dict): keywords for line noise filtering. See
            the documentation for the notch_filter functin in MNE 
            (https://mne.tools/stable/generated/mne.filter.notch_filter.html)
            for further details. Default is an empty dict.
        '''
        info = mne.create_info(ch_names=self.ch_names, 
                               sfreq=self.sfreq,
                               ch_types=self.ch_types)
        montage = mne.channels.make_standard_montage(kind=self.montage_kind)
        prep_params = {
            'ref_chs' : ref_chs,
            'reref_chs' : reref_chs,
            'line_freqs' : np.arange(line_freq, self.sfreq / 2, line_freq),
            'max_iterations' : max_prep_iter
        }
        self._crop()
        np_eeg = self.epoch[:, 2:-1].T
        mne_eeg = mne.io.RawArray(np_eeg, info)

        try:
            prep = pyprep.PrepPipeline(mne_eeg, prep_params, montage, 
                                       filter_kwargs=filter_kwargs,
                                       ransac=True)
            prep.fit()
            self.prep = prep
        except(OSError, ValueError):
            self.bad = 'ransac_event'
            prep = pyprep.PrepPipeline(mne_eeg, prep_params, montage, 
                                       filter_kwargs=filter_kwargs,
                                       ransac=False)
            try:
                prep.fit()
                self.prep = prep
            except(ValueError):
                self.bad = 'robust_event'

        if self.bad != 'robust_event':
            self._fit_whole(info, montage, prep_params, filter_kwargs)
    
    def report(self):
        '''
        Return a report (dict) on bad channels and PREP parameters. 
        '''
        data = self.prep
        if data != None:
            try:
                bad_report = {
                    'interpolated_channels' : data.interpolated_channels,
                    'noisy_channels_after_interpolation' : data.noisy_channels_after_interpolation,
                    'noisy_channels_before_interpolation' : data.noisy_channels_before_interpolation,
                    'still_noisy_channels' : data.still_noisy_channels,
                    'prep_params' : data.prep_params,
                    'ransac_settings' : data.ransac_settings,
                    'bad_status' : self.bad
                }
            except:
                bad_report = {
                    'prep_params' : data.prep_params,
                    'ransac_settings' : data.ransac_settings,
                    'bad_status' : self.bad
                }
        else:
            bad_report = None
        return bad_report

    def save_numpy(self, filename=None):
        '''
        Save cleaned EEG epoch as a numpy array.

        Parameter:
            - filename (str): name of the file. Default is None, which entails
            that the newly created file is stored as 
            ./numpy/{original_mat_file_name}.npy
        
        Return: none.
        '''
        if self.bad != 'robust_whole':
            np_cleaned = self.prep.EEG_clean
            if filename == None:
                parent_dir = os.path.abspath(os.path.join(self.mat_dir, os.pardir))
                filename = parent_dir + os.sep + 'numpy' + os.sep + self.name + '.npy'
            with open(filename, 'wb') as f:
                np.save(f, np_cleaned)
        else:
            print('Data too bad to use PREP, therefore no output numpy file.')