import numpy as np

import matplotlib.pyplot as plt
from random import sample, seed

from load_data import read_data, check_timeseries_length

seed(1)

aeTrain = np.genfromtxt("resources/ae.train", dtype=None)
aeTest = np.genfromtxt("resources/ae.test", dtype=None)

train_lengths, train_maxy = check_timeseries_length(aeTrain)
train_input, train_output = read_data(aeTrain, train_lengths, 270, train_maxy)

third_speaker_samples = sample(range(30, 60), 3)
sixth_speaker_samples = sample(range(150, 180), 3)

speaker = "Third"
sample_no = 1
for subplot_id, idx in enumerate([*third_speaker_samples, *sixth_speaker_samples]):
    if sample_no > 3:
        speaker = "Sixth"
        sample_no = 1
    plt.subplot(2, 3, subplot_id+1)
    for channel in range(12):
        y = np.trim_zeros(train_input[idx, :, channel])
        plt.plot(y)
    plt.xticks(np.arange(0, 30, 5))
    plt.title(speaker + " speaker, sample no " + str(sample_no))
    sample_no += 1

plt.show()
