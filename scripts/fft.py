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

subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
str_sbj = [f"Pessoa {subject}" for subject in subjects]
results = []
for subject in subjects:
    # print(f"subject: {subject}")
    results.append(fft_subject_predict(subject))
df = pd.DataFrame(results, columns=["Acurácia (%)", "Erro padrão"], index=str_sbj)