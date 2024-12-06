
import multiprocessing as mp
from asrpy import ASR
import pandas as pd
import numpy as np
import pyxdf
import mne
import json
import sys
import os
import gc

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()


def get_streams(data):

    TRIGGERS = [
    'BaseLineBeginTimeStamp',
    'BaseLineEndTimeStamp',
    'CueTimeStamp',
    'CueDisappearedTimeStamp',
    'ObjectShownTimeStamp',
    'BeepPlayedTimeStamp',
    'ButtonPressedTimeStamp'
    ]

    TRIAL_INFO = 'TrialInformationInt'

    EEG_SIGNAL = 'openvibeSignal'

    ET_SIGNAL = 'EyeTrackingGazeHMDFloat'

    STIMULUS = ['ToolCueOrientationString', 'ToolCueOrientationInt']

    TOOLS = ['daisygrubber', 'fishscaler', 'flowercutter', 'fork', 'paintbrush',
           'paletteknife', 'screwdriver', 'spatula', 'spokewrench', 'trowel',
           'wrench', 'zester']

    FAMILIAR = ['fork','trowel','screwdriver','spatula','paintbrush','wrench']

    tool_fam_map = {t:('fam' if t in FAMILIAR else 'unfam') for t in TOOLS}

    EVENT_MAPPING = {'task': 1,
                     'inter-task-object': 2,
                     'object': 3,
                     'action': 4,
                     'inter-trial':5 }

    streams = {}

    if len(data) == 0:
        print('No data in stream')
        return 0
    for i in range(len(data)):
        if data[i]["info"]['name'][0] == EEG_SIGNAL:
            streams['eeg'] = int(i)

        elif data[i]["info"]['name'][0] == ET_SIGNAL:
            streams['et'] = int(i)

        elif data[i]["info"]['name'][0] in TRIAL_INFO:
            streams[data[i]["info"]['name'][0]] = int(i)

        elif data[i]["info"]['name'][0] in TRIGGERS:
            streams[data[i]["info"]['name'][0]] = int(i)

        else:
            continue

    return streams

def read_stream_info(f):
    data, header = pyxdf.load_xdf(f)
    stream = pd.Series(get_streams(data))

    stream['subject_idect_id'] = f.split('\\')[1].split('_')[0]
    stream['block_id'] = f.split('\\')[2].split('_')[1].split('.')[0]

#     try:

    stream['baseline_check'] = np.mean(
        data[stream['BaseLineEndTimeStamp']]['time_stamps']
        - data[stream['BaseLineBeginTimeStamp']]['time_stamps']
    )

    stream['cue_dur'] = np.mean(
        data[stream['CueDisappearedTimeStamp']]['time_stamps']
        - data[stream['CueTimeStamp']]['time_stamps']
    )

    stream['break_dur'] = np.mean(
        data[int(stream['ObjectShownTimeStamp'])]['time_stamps']
        - data[int(stream['CueDisappearedTimeStamp'])]['time_stamps']
    )

    stream['object_dur'] = np.mean(
        data[int(stream['BeepPlayedTimeStamp'])]['time_stamps']
        - data[int(stream['ObjectShownTimeStamp'])]['time_stamps']
    )

    stream['action_dur'] = np.mean(
        data[int(stream['ButtonPressedTimeStamp'])]['time_stamps']
        - data[int(stream['BeepPlayedTimeStamp'])]['time_stamps']
    )

    stream['et_frame_rates'] = 1/np.mean(
        data[int(stream['et'])]['time_stamps'][1:]
        - data[int(stream['et'])]['time_stamps'][:-1]
    )

    if 'eeg' not in stream:
        stream['eeg_frame_rates'] = np.nan
    else:

        stream['eeg_frame_rates'] = 1/np.mean(
            data[stream['eeg']]['time_stamps'][1:]
            - data[stream['eeg']]['time_stamps'][:-1]
        )

    return pd.DataFrame(stream).transpose()


def read_et_df(f, tool_dict=None, fam_dict=None, task_dict=None, ori_dict=None):
    print(f)

    try:
        data, header = pyxdf.load_xdf(f)
        stream = pd.Series(get_streams(data))

        stream['subject_idect_id'] = f.split('\\')[1].split('_')[0]
        stream['block_id'] = f.split('\\')[2].split('_')[1].split('.')[0]

        et_tmp = pd.DataFrame(
            data=data[stream['et']]['time_series'],
            index=data[stream['et']]['time_stamps'],
            columns=list(data[stream['et']]['info']['desc'][0].keys())
        )

        et_tmp['timestamps'] = data[stream['et']]['time_stamps']
        et_tmp['subject_idect_id'] = stream.subject_idect_id
        et_tmp['block_id'] = stream.block_id

        trial_info = pd.DataFrame(
            data=data[stream['TrialInformationInt']]['time_series'],
            index=data[stream['TrialInformationInt']]['time_stamps'],
            columns=list(data[stream['TrialInformationInt']]['info']['desc'][0].keys())
        )

        trial_info['beep'] = data[stream['BeepPlayedTimeStamp']]['time_stamps']
        trial_info['cue'] = data[stream['CueTimeStamp']]['time_stamps']
        trial_info['object'] = data[stream['ObjectShownTimeStamp']]['time_stamps']

        et_block = pd.DataFrame()
        for idx, trial in trial_info.iterrows():
            t = et_tmp.loc[trial.cue - 0.1 : trial.beep + 1].copy()
            t.reset_index()
            t['trial_id'] = trial.TrialNumber
            t['tool_id'] = trial.ToolId
            t['cue_time'] = trial.cue
            t['beep_time'] = trial.beep
            t['object_time'] = trial.object
            t['tool_name'] = tool_dict.loc[trial.ToolId, 'tool_name']
            t['tool_pos_x'] = tool_dict.loc[trial.ToolId, 'tool_pos_x']
            t['tool_pos_y'] = tool_dict.loc[trial.ToolId, 'tool_pos_y']
            t['tool_pos_z'] = tool_dict.loc[trial.ToolId, 'tool_pos_z']


            t['task'] = trial.Task
            t['orientation'] = trial.Orientation
            t['time'] = t.timestamps - trial.object
            t['task'] = t.task.map(task_dict)
            t['orientation'] = t.orientation.map(ori_dict)
            t['familiarity'] = t.tool_name.map(fam_dict)

            et_block=pd.concat([et_block, t], ignore_index=True)
    except ValueError as ve:
        print(f'error in file: {f}. ######### {ve} ')
        return []


    print(et_block.head())

    return et_block


def get_channel_names(temp, n, montagefile):

    with open(montagefile) as f: #open the montage file provided by Debbie
        montage = json.loads(f.read())

    montage['ch-1']='Fp1'
    montage['ch-2']='Fpz'
    montage['ch-3']='Fp2'
    ch_names = []
    for i in range(n):
        n = temp[i]['label'][0].replace("ExG", "ch-")
        ch_names.append(montage[n] if "ch-" in n else n)  # only eeg channels are mapped, the aux channels are not
    return ch_names


# save files
def save_file(f, ftype, subject_id, block_id, save_dir):

    dir_structure = {
    'baseline': "eeg/00_ASR_baseline",
    'events': "eeg/01_events_added",
    'filtered': "eeg/02_filtered",
    'epoched': "eeg/03_epoched",
    'cleaned': "eeg/04_cleaned",
    'rawepochs': "eeg/05_rawepochs"
    }
    block_id = str(block_id)

    save_dir_name = os.path.join(save_dir, subject_id, block_id, dir_structure[ftype])

    if not os.path.isdir(save_dir_name):
        os.makedirs(save_dir_name)

    if ftype in ['events', 'filtered', 'cleaned', 'baseline']:
        fend = '_eeg.fif'
    else:
        fend = '_epo.fif'

    fname = f"block_{str(block_id)}{fend}"

    f.save(os.path.join(save_dir_name, fname), overwrite=True)

    # when saving the epoched mne file also save a numpy version
    if ftype == 'epoched':
        epoch_array = f.get_data()  # convert mne to numpy array
        fname = f"block_{block_id}_epo.npy"
        with open(os.path.join(save_dir_name, fname), 'wb') as f:
            np.save(f, epoch_array)

    return save_dir_name


def get_eeg_baseline(subject_id='', montagefile='', save_dir=''):

    fname = f'./raw_data/{subject_id}/{subject_id}_1.xdf'

    try:
        data, header = pyxdf.load_xdf(fname)

        # get stream names and index in data
        stream = pd.Series(get_streams(data))

        begin = int(np.searchsorted(
            data[stream['eeg']]["time_stamps"],
            data[stream['BaseLineBeginTimeStamp']]["time_stamps"]
        ))

        end = int(np.searchsorted(
                    data[stream['eeg']]["time_stamps"],
                    data[stream['BaseLineEndTimeStamp']]["time_stamps"]
        ))

        start_time = data[stream['eeg']]["time_stamps"][0]
        eeg_time = data[stream['eeg']]["time_stamps"] - start_time  # get eeg time points

        begin_time = data[stream['eeg']]["time_stamps"][begin] - start_time
        end_time = data[stream['eeg']]["time_stamps"][end] - start_time

        sr = data[stream['eeg']]['info']['effective_srate']  # sampling rate of eeg

        n_ch = int(data[stream['eeg']]['info']['channel_count'][0])  # no. of eeg channels

        ch_typ = ['eeg'] * n_ch  # set channel type for mne

        ch_names = get_channel_names(
            data[stream['eeg']]['info']['desc'][0]['channels'][0]['channel'],
            n_ch,
            montagefile
        )
        # create mne info
        info = mne.create_info(ch_names, ch_types=ch_typ, sfreq=sr)

        # mne data array should be in (nChannel,nSamples) whereas xdf stores in (nSamples,nChannel)
        eeg = mne.io.RawArray(np.transpose(data[stream['eeg']]["time_series"]), info)

        # drop auxiliaries and not needed channels
        eeg.drop_channels(['BIP65', 'BIP66', 'BIP67', 'BIP68', 'AUX69', 'AUX70', 'AUX71', 'AUX72'])

        # set the montage
        eeg.set_montage('standard_1020')

        # reject the first 20 secs to get rid of voltage swings at the start
        # resample to 256Hz
        # Note: the raw and the event are resampled simultaneously so that they stay more or less in synch.
        eeg.crop(tmin=begin_time, tmax=end_time)

        eeg_resamp = eeg.resample(sfreq=256,)

        del eeg
        gc.collect()

        # EEG is recorded with average reference. Re-refer to Cz
        eeg_resamp.set_eeg_reference(ref_channels=['Cz'])

        # Filter resampled EEG
        # High pass 0.1 Hz
        # Low pass 120 Hz
        # notch 50Hz,100Hz
        eeg_resamp.filter(l_freq=2, h_freq=50)
        eeg_resamp.notch_filter(freqs=[50, 100])
        # save to filtered folder
        save_file(eeg_resamp, 'baseline', subject_id, "1", save_dir)

    except Exception as e:
        print(e)

    return eeg_resamp




def get_eeg_and_preprocess(file_name, trial_info=None, montage='', save_dir=''):

    # get subject_id and block_id from filename
    subject_id = file_name.split('/')[3].split('_')[0]

    block_id = int(file_name.split('/')[3].split('_')[1].split('.')[0])

    print(subject_id, block_id)
    # flag to keep track of data status
    out = ''

    s_name = {}

    try:
        # read xdf file
        data, header = pyxdf.load_xdf(file_name)

        # get stream names and index in data
        stream = pd.Series(get_streams(data))

        # choose only event streams not et or eeg data
        event_streams = stream[[k for k in stream.keys() if k not in ['et', 'eeg']]]
           #  EVENTS STREAM NAMES: ['BeepPlayedTimeStamp', 'CueTimeStamp', 'TrialInformationInt',
           #  'BaseLineBeginTimeStamp', 'BaseLineEndTimeStamp', 'ButtonPressedTimeStamp', 'ObjectShownTimeStamp',
           # 'CueDisappearedTimeStamp']
        if 'BaseLineBeginTimeStamp' not in stream.keys():
            print('Baseline timestamp not found')

        if 99 in list(event_streams.values):
            print(" EVENTS STREAMS not found. Skipped")
            out = 'skipped'
        else:
            start_time = data[stream['eeg']]["time_stamps"][0]
            eeg_time = data[stream['eeg']]["time_stamps"] - start_time  # get eeg time points

            begin = int(np.searchsorted(
                data[stream['eeg']]["time_stamps"],
                data[stream['BaseLineBeginTimeStamp']]["time_stamps"][0]
            ))

            end = int(np.searchsorted(
                        data[stream['eeg']]["time_stamps"],
                        data[stream['BaseLineEndTimeStamp']]["time_stamps"][0]
            ))

            begin_time = data[stream['eeg']]["time_stamps"][begin] - start_time
            end_time = data[stream['eeg']]["time_stamps"][end] - start_time

            sr = data[stream['eeg']]['info']['effective_srate']  # sampling rate of eeg

            n_ch = int(data[stream['eeg']]['info']['channel_count'][0])  # no. of eeg channels

            ch_typ = ['eeg'] * n_ch  # set channel type for mne

            ch_names = get_channel_names(
                data[stream['eeg']]['info']['desc'][0]['channels'][0]['channel'],
                n_ch,
                montage
            )

            # map the trial start cue i.e. the first index in the event_streams to ecent array for mne
            tdf = trial_info.query('subject_id==@subject_id and block_id==@block_id')
            # print(tdf)
            trial_event_map = {t:i+1 for i,t in enumerate(trial_info.condition.unique())}

            tdf['int_cond'] = tdf.condition.map(trial_event_map)
            # print(tdf)
            # find events timestamps within eeg data stream
            x = np.searchsorted(
                data[stream['eeg']]["time_stamps"],
                data[stream['ObjectShownTimeStamp']]["time_stamps"]
            )
            # print(x.shape,)


            event_trial_start = np.zeros((x.shape[0], 3), dtype=int)
            event_trial_start[:, 0] = x
            event_trial_start[:, 2] = tdf.int_cond.values #.to_numpy(dtype=int)

            del trial_info, tdf
            gc.collect()

            # create mne info
            info = mne.create_info(ch_names, ch_types=ch_typ, sfreq=sr)

            # mne data array should be in (nChannel,nSamples) whereas xdf stores in (nSamples,nChannel)
            eeg = mne.io.RawArray(np.transpose(data[stream['eeg']]["time_series"]), info)

            # drop auxiliaries and not needed channels
            eeg.drop_channels(['BIP65', 'BIP66', 'BIP67', 'BIP68', 'AUX69', 'AUX70', 'AUX71', 'AUX72'])

            # set the montage
            eeg.set_montage('standard_1020')

            # reject the first 20 secs to get rid of voltage swings at the start
            # resample to 256Hz
            # Note: the raw and the event are resampled simultaneously so that they stay more or less in synch.
            eeg.crop(tmin=15)
            eeg_resamp, events_resamp = eeg.resample(sfreq=256, events=event_trial_start)

            del eeg
            gc.collect()

            # EEG is recorded with average reference. Re-refer to Cz
            eeg_resamp.set_eeg_reference(ref_channels=['Cz'])
            # eeg_resamp.drop_channels(['Cz',])
              # save to events folder
            s_name['events'] = save_file(eeg_resamp, 'events', subject_id, block_id, save_dir)

            # Filter resampled EEG
            # High pass 0.1 Hz
            # Low pass 120 Hz
            # notch 50Hz,100Hz
            eeg_resamp.filter(l_freq=0.2, h_freq=120, h_trans_bandwidth=1.0, verbose=False)
            eeg_resamp.notch_filter(freqs=[16.667, 50, 100], verbose=False)
            # save to filtered folder
            s_name['filtered'] = save_file(eeg_resamp, 'filtered', subject_id, block_id, save_dir)

            # epochs paramaters
            tmin = -0.1  # 100 msec before the event boundary
            tmax = 3.1  # each trial is 2.0 + 0.5 + 3.0 + 0.5...NOTE: first 0.5 sec of action is included
            baseline = (tmin, 0)  # i.e. the entire zone from tmin to 0
            epochs = mne.Epochs(eeg_resamp, events_resamp, event_id=trial_event_map,
                tmin=tmin, tmax=tmax, baseline=None, detrend=0, verbose=False)
            # epochs.drop_bad()
            # save to rawepochs folder
            s_name['rawepochs'] = save_file(epochs, 'rawepochs', subject_id, block_id, save_dir)

            del epochs
            gc.collect()

            # run asr to clean the filtered raw data
            # Apply the ASR
            asr_baseline = eeg_resamp.copy()
            asr = ASR(sfreq=eeg_resamp.info["sfreq"], cutoff=5)
            asr_baseline.crop(tmin=begin_time+1, tmax=end_time-1)
            print(eeg_resamp.get_data().shape)

            save_file(asr_baseline, 'baseline', subject_id, block_id, save_dir)
            asr.fit(eeg_resamp)

            eeg_resamp = asr.transform(eeg_resamp)

            print("ASR fitted\n")
            # save to cleaned folder
            s_name['cleaned'] = save_file(eeg_resamp, 'cleaned', subject_id, block_id, save_dir)

            # epoch the cleaned eeg, use the parameter already set-up
            clean_epochs = mne.Epochs(eeg_resamp, events_resamp, event_id=trial_event_map, tmin=tmin, tmax=tmax, baseline=None, detrend=0)
            # clean_epochs.drop_bad()
            # save to epoched folder
            s_name['epoched'] = save_file(clean_epochs, 'epoched', subject_id, block_id, save_dir)

            del clean_epochs, eeg_resamp
            gc.collect()

            out = "success"

    except Exception as e:
        print(e)
        out = 'failed'
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[-1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return pd.Series({'subject_id': subject_id, 'block_id':block_id,
        'status': out, 'save_dirs': s_name, 'error': e, 'line_no': exc_tb.tb_lineno})

    return pd.Series({'subject_id': subject_id, 'block_id':block_id, 'status': out, 'save_dirs': s_name})
