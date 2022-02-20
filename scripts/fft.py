import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mne import read_events
from mne.io import Raw
from numpy.fft import fft, fftfreq
from scipy.signal import detrend, butter, sosfilt

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


def get_subject_data(*, subject: int = 12, run: int= 5):
    raw_path = f'/home/lucas-c/workspace/databases/ssvep_exo/subject{subject}_run{run}_raw.fif'
    events_path =f'/home/lucas-c/workspace/databases/ssvep_exo/subject{subject}_run{run}_eve.fif'

    raw = Raw(raw_path, preload=True, verbose=False)
    events = pd.DataFrame(read_events(events_path), columns=["sample", "junk", "label"])
    events = events.drop(columns="junk")
    return raw.get_data(), events


def plot_label_data_for_channel(label:str, channel: int, *, filtered = True) -> None:

    print(channel)
    record_data, events = get_subject_data(subject=12, run=5)

    sample_rate = 256 #hz
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
                sos = butter(5, (freq - 0.3, freq + 0.3), 'bp', fs=sample_rate, output='sos')
                filt_data = sosfilt(sos, data)

                freqs = abs(fftfreq(len(filt_data), 1/sample_rate))
                fft_data = abs(fft(filt_data))/len(filt_data)

                freq_index = np.where(freqs == freq)[0][0]
                magnitude = fft_data[freq_index]
                # print(magnitude)
                
                
                plt.xlim([0, 30])
                plt.plot(freqs, fft_data)
        else: 
            freqs = abs(fftfreq(len(data), 1/sample_rate))
            fft_data = abs(fft(data))/len(data)

            freq_index = np.where(freqs == 13)[0][0]
            magnitude = fft_data[freq_index]
            # print(magnitude)
            
            
            plt.xlim([0, 50])
            plt.legend()
            plt.plot(freqs, fft_data, linewidth= 0.5)

        break

label = "17hz"
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.title(f"Canal {CHANNELS[i]}")
    plot_label_data_for_channel(label, i, filtered=False)

# plt.suptitle(f"Led Frequency {label}")
plt.show()
