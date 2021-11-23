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
    def __init__(self, mat_dir, 
                event='CLOSED EYES', 
                ch_names=['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 
                          'T8', 'F8', 'AF4', 'FC6', 'F4'],
                ch_types='eeg',
                sfreq=256,
                montage_kind='standard_1020'):
        self.mat = _loadtmat(mat_dir)
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

    def _find_mask(self):
        event_split = self.event.upper().split()
        event_id = None
        for i, event_struct in enumerate(self.events):
            desc = event_struct[-1].__dict__.values()
            if sum([x in desc for x in event_split]) == len(event_split):
                event_id = i
        self.mask = np.argwhere((self.record[:, -1] >= self.events[event_id][0]) \
                         & (self.record[:, -1] <= self.events[event_id][1]))
        return self.mask
    
    def _crop(self):
        if self.event != None:
            self.epoch = self.recording[self._find_mask()]
        else:
            self.epoch = self.recording

    def _fit_whole(self, info, montage, prep_params, filter_kwargs):
        np_eeg = self.recording.squeeze(axis=1)[:, 2:-1].T
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
                self.prep.EEG_clean = self.prep.EEG_clean[self.mask]
            except(ValueError):
                self.bad = 'robust_whole'
                raise ValueError('Data is too bad to perform PREP')
        self.prep = prep
        self.prep.EEG_clean = self.prep.EEG_clean[self.mask]

    def fit(self, 
            ch_types='eeg',
            ref_chs='eeg',
            reref_chs='eeg',
            line_freq=60,
            max_prep_iter=4,
            filter_kwargs={}):
        info = mne.create_info(ch_names=self.ch_names, 
                               sfreq=self.sfreq,
                               ch_types=self.ch_types)
        montage = mne.channels.make_standard_montage(kind=self.montage_kind)
        prep_params = {
            'ref_chs' : ref_chs,
            'reref_chs' : reref_chs,
            'line_freqs' : np.arange(line_freq, sfreq / 2, line_freq),
            'max_iterations' : max_prep_iter
        }
        self._crop()
        np_eeg = self.epoch.squeeze(axis=1)[:, 2:-1].T
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
                    'ransac_settings' : data.ransac_settings
                    'bad_status' : self.bad
                }
        else:
            bad_report = None
        return bad_report

    def save_numpy(self):
        if self.bad != 'robust_whole':
            np_cleaned = self.prep.EEG_clean
            parent_dir = os.path.abspath(os.path.join(self.mat_dir, os.pardir))
            new_dir = parent_dir + os.sep + 'numpy' + os.sep + self.name + '.npy'
            with open(new_dir, 'wb') as f:
                np.save(f, np_cleaned)
        else:
            print('Data too bad to use PREP, therefore no output numpy file.')
