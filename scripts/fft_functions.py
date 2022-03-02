from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from mne import read_events
from mne.io import Raw
from numpy.fft import fft, fftfreq
from scipy.signal import detrend, butter, sosfilt


SAMPLE_RATE = 256 #Hz

LABELS = {
    "rest": 0,
    "13hz": 1,
    "17hz": 3,
    "21hz": 2
}

CHANNELS = {
    0: "Oz",
    1: "O1",
    2: "O2",
    3: "POz",
    4: "PO3",
    5: "PO4",
    6: "PO7",
    7: "PO8",
}

def get_subject_data(subject: int = 12, run: int= 5):
    subject_str = "0"+str(subject) if subject < 10 else str(subject)
    raw_path = f'/home/lucas-c/workspace/databases/ssvep_exo/subject{subject_str}_run{run}_raw.fif'
    events_path =f'/home/lucas-c/workspace/databases/ssvep_exo/subject{subject_str}_run{run}_eve.fif'

    raw = Raw(raw_path, preload=True, verbose=False)
    events = pd.DataFrame(read_events(events_path), columns=["sample", "junk", "label"])
    events = events.drop(columns="junk")
    return raw.get_data(), events

def plot_fft_data(data):
    freqs = abs(fftfreq(len(data), 1/SAMPLE_RATE))
    fft_data = abs(fft(data))/len(data)

    plt.xlim([0, 50])
    plt.legend()
    plt.plot(freqs, fft_data, linewidth= 0.5)


def plot_label_data_for_channel(label:str, channel: int, *, filtered = True) -> None:

    print(channel)
    record_data, events = get_subject_data(subject=12, run=5)

    channel_data = record_data[channel]

    plt.axvline(x=13, color="r", linestyle='--', dashes=(5, 1), label="13Hz")
    plt.axvline(x=17, color="b", linestyle='--', dashes=(5, 5), label="17Hz")
    plt.axvline(x=21, color="g", linestyle='--', dashes=(5, 10), label="21Hz")

    samples = events.loc[events["label"] == LABELS[label], "sample"]
    step = events["sample"].iloc[2] - events["sample"].iloc[1]
    for sample in samples.values:
        data = detrend(channel_data[sample: sample + step])

        if filtered:
            for freq in [13, 17, 21]:
                sos = butter(5, (freq - 0.3, freq + 0.3), 'bp', fs=SAMPLE_RATE, output='sos')
                filt_data = sosfilt(sos, data)

                plot_fft_data(filt_data)
        else: 
            plot_fft_data(data)

        break

def estimate_epoch_result(epoch_data, filtered = True):
    max_mag = -1
    for freq in [13, 17, 21]:
        if filtered:
            sos = butter(5, (freq - 0.3, freq + 0.3), 'bp', fs=SAMPLE_RATE, output='sos')
            data = sosfilt(sos, epoch_data)
        else:
            data = epoch_data

        freqs = abs(fftfreq(len(data), 1/SAMPLE_RATE))
        fft_data = abs(fft(data))/len(data)

        freq_idx = (np.abs(freqs - freq)).argmin()
        magnitude = max(fft_data[freq_idx - 2 : freq_idx + 2])

        if magnitude > max_mag:
            max_mag = magnitude
            result = LABELS[f"{freq}hz"]

    return result


def fft_predict(subject=12, run=5, *, filtered = True, compare_to="result"):

    record_data, events = get_subject_data(subject, run)

    results = pd.DataFrame(columns=["sample", *CHANNELS.values()])
    results["sample"] = events["sample"]
    results["label"] = events["label"]
    step = results["sample"].iloc[2] - results["sample"].iloc[1]
    results.set_index('sample', inplace=True)

    for i, channel in CHANNELS.items():
        channel_data = record_data[i]
        for sample in results.index:
            try:
                epoch_data = detrend(channel_data[sample: sample + step])
                results.loc[sample, channel] = estimate_epoch_result(epoch_data, filtered)
            except:
                results.loc[sample, channel] = np.nan
        
    results["result"] = results.mode(axis=1)[0]
    results = results.drop(results[results["label"] == 4].index).dropna()
    return 100 * sum(results["label"] == results[compare_to])/len(results)

def fft_subject_predict(subject, filtered=True):
    subject_str = "0"+str(subject) if subject < 10 else str(subject)
    path = f'/home/lucas-c/workspace/databases/ssvep_exo/subject{subject_str}_**'
    n_records = len(glob.glob(path))//2
    scores = []
    for run in range(1, n_records+1):
        scores.append(fft_predict(subject, run, filtered=filtered))

    score = round(np.mean(scores), 2)
    error = round(np.std(scores, ddof=1) / np.sqrt(np.size(scores)),2)
    return score, error
    # print(f"accuracy: {}%")
    # print(f"std error: {}")

    