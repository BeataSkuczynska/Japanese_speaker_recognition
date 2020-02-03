import numpy as np


def check_timeseries_length(raw_inputs):
    longest = 0
    length = 0
    lengths = []
    for row in raw_inputs:
        if row[11] == 1.0:
            if length > longest:
                longest = length
            lengths.append(length)
            length = 0
        else:
            length += 1
    return lengths, longest


def read_data(raw_inputs, lengths, xdim, ydim, zdim=12, test=False):

    test_block_lengths = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    chunked_inputs = np.zeros(shape=(xdim, ydim, zdim))
    outputs = np.zeros(shape=(xdim, ydim, 9))
    readindex = 0
    speaker_idx = 0
    block_counter = 0
    for idx, length in enumerate(lengths):
        chunked_inputs[idx, 0:length] = raw_inputs[readindex:readindex+length]
        readindex += length + 1

        teacher = np.zeros(shape=(ydim, 9))
        if test:
            if block_counter == test_block_lengths[speaker_idx]:
                speaker_idx += 1
                block_counter = 1
            else:
                block_counter += 1
        else:
            speaker_idx = max(0, int(np.ceil(idx / 30))-1)
        teacher[:length, speaker_idx] = np.ones(shape=length)
        outputs[idx] = teacher

    return chunked_inputs, outputs



