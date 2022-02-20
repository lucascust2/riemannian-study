import matplotlib.pyplot as plt

from fft_functions import *


label = "17hz"

# Plot Channels of one time window
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.title(f"Canal {CHANNELS[i]}")
#     plot_label_data_for_channel(label, i, filtered=True)
# plt.suptitle(f"Led Frequency {label}")
# plt.show()

fft_predict()