import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from mne import read_events
from mne.io import Raw
from numpy.fft import fft, fftfreq
from scipy.signal import detrend

def get_subject_data(*, subject: int, run: int):
    # paths = glob("./exo_data/**")
    raw_path = f'./exo_data/subject{subject}_run{run}_raw.fif'
    events_path = f'./exo_data/subject{subject}_run{run}_events.fif'

    raw = Raw(raw_path, preload=True, verbose=False)
    events = pd.DataFrame(read_events(events_path),columns=["sample", "junk", "label"])
    return raw.get_data(), events


record_data, events = get_subject_data(subject=12, run=5)

for channel in record_data:

    
    freqs = abs(fftfreq(len(channel), 1/20000))

    fftdata = abs(fft(channel))/len(channel)

    plt.plot(freqs, fftdata)

    

plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude")
plt.show()
