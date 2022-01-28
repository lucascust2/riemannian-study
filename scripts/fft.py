import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from mne import read_events
from mne.io import Raw
from numpy.fft import fft, fftfreq
from scipy.signal import detrend

LABELS = {
    "rest": 0,
    "13hz": 1,
    "17hz": 3,
    "21hz": 2
}

def get_subject_data(*, subject: int, run: int):
    raw_path = f'./exo_data/subject{subject}_run{run}_raw.fif'
    events_path = f'./exo_data/subject{subject}_run{run}_eve.fif'

    raw = Raw(raw_path, preload=True, verbose=False)
    events = pd.DataFrame(read_events(events_path), columns=["sample", "junk", "label"])
    events = events.drop(columns="junk")
    return raw.get_data(), events


def plot_label_data_for_channel(label:str, channel) -> None:

    record_data, events = get_subject_data(subject=12, run=5)

    sample_rate = 256 #hz
    channel_data = record_data[channel]
    for freq in [13, 17, 21]:
        plt.axvline(x=freq, color="r")

    samples = events.loc[events["label"] == LABELS[label], "sample"]
    step = events["sample"].iloc[2] - events["sample"].iloc[1]
    for sample in samples.values:
        data = detrend(channel_data[sample: sample + step])

        freqs = abs(fftfreq(len(data), 1/sample_rate))
        fftdata = abs(fft(data))/len(data)

        plt.plot(freqs, fftdata)
        break

label = "17hz"
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.title(f"Channel {i+1}")
    plot_label_data_for_channel(label, i)

plt.suptitle(f"Led Frequency {label}")
plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude")
plt.show()
