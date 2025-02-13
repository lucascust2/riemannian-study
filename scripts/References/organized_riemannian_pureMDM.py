"""
====================================================================
Offline SSVEP-based BCI Multiclass Prediction
====================================================================
"""

# generic import
import numpy as np
from numpy import array, empty, ndarray, tile

# gzip
import gzip

# mne import
from mne import create_info
from mne.io import RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# lib to save the trained model
import pickle
import random
from time import time


def _bandpass_filter(signal, lowcut, highcut):
    """Bandpass filter using MNE"""
    return signal.copy().filter(l_freq=lowcut, h_freq=highcut, method="iir").get_data()


###############################################################################
# Loading EEG data


with gzip.open("./data/old_data/record-[2012.07.06-19.06.14].pz", "rb") as f:
    o = pickle.load(f, encoding="latin1")
raw_signal = o["raw_signal"].T
event_pos = o["event_pos"].reshape((o["event_pos"].shape[0]))
event_type = o["event_type"].reshape((o["event_type"].shape[0]))
sfreq = 256
classes = ["Resting", "13Hz", "21Hz", "17Hz"]
channels = array(["Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4"])

###############################################################################
t1 = time()
# get label
labels = list()
for e in event_type:
    if e == 33024:
        labels.append("Resting")
    elif e == 33025:
        labels.append("13Hz")
    elif e == 33026:
        labels.append("21Hz")
    elif e == 33027:
        labels.append("17Hz")
labels = list(labels)


################################################################################
info = create_info(
    ch_names=["Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4"],
    ch_types=["eeg"] * 8,
    sfreq=sfreq,
)

raw_ext = RawArray(raw_signal, info, verbose=False)
################################################################################

frequencies = [13.0, 17.0, 21.0]
freq_band = 0.1
ext_signal = np.vstack(
    [
        _bandpass_filter(
            raw_ext,
            lowcut=f - freq_band,
            highcut=f + freq_band,
        )
        for f in frequencies
    ]
)
# ext_signal = ext_signal[1:,:]

###############################################################################
ext_trials: ndarray = np.array([])
for e, t in zip(event_type, event_pos):
    if e == 32779:  # start of a trial
        start = t + 2 * sfreq
        stop = t + 5 * sfreq
        np.append(ext_trials, ext_signal[:, start:stop])
ext_trials = array(ext_trials)
ext_trials = ext_trials - tile(
    ext_trials.mean(axis=2).reshape(ext_trials.shape[0], ext_trials.shape[1], 1),
    (1, 1, ext_trials.shape[2]),
)


cov_ext_trials = Covariances(estimator="lwf").transform(ext_trials)

i = 0
y_train: ndarray = np.array([])
n = len(labels) * 0.7
n = int(n)
cov_train = empty((n, 24, 24))
while i < n:

    val = random.randint(0, len(labels) - 1)

    y_train = np.append(y_train, labels[val])
    new_labels = np.delete(labels, val)

    cov_train[i, :, :] = cov_ext_trials[val, :, :]
    cov_ext_trials = np.delete(cov_ext_trials, val, axis=0)

    i += 1

y_test = new_labels
cov_test = cov_ext_trials

# y_train = labels[::2] # take even indexes
# y_test = labels[1::2] # take odd indexes

# cov_train = cov_ext_trials[::2]
# cov_test = cov_ext_trials[1::2]


cov_centers = empty((len(classes), 24, 24))

for i, l in enumerate(classes):
    cov_centers[i, :, :] = mean_riemann(cov_train[y_train == l, :, :])


## Mean calculated with Affine-Invariant Riemannian metric (AIRM)
## implemented in pyRiemann
accuracy = list()
for sample, labels in zip(cov_test, y_test):
    dist = [distance_riemann(sample, cov_centers[m]) for m in range(len(classes))]
    if classes[array(dist).argmin()] == labels:
        accuracy.append(1)
    else:
        accuracy.append(0)
test_accuracy = 100.0 * array(accuracy).sum() / len(y_test)

t2 = time()
print("tempo: " + str(t2 - t1))
print("Evaluation accuracy on test set is %.2f%%" % test_accuracy)
