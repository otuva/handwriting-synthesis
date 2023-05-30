from __future__ import print_function

import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import processed_data_path
from handwriting_synthesis.training.preparation import get_stroke_sequence, collect_data


def prepare():
    print('traversing data directory...')
    stroke_fnames, transcriptions, writer_ids = collect_data()

    print('dumping to numpy arrays...')
    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
    w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
    valid_mask = np.zeros([len(stroke_fnames)], dtype=np.bool)

    for i, (stroke_fname, c_i, w_id_i) in enumerate(zip(stroke_fnames, transcriptions, writer_ids)):
        if i % 200 == 0:
            print(i, '\t', '/', len(stroke_fnames))
        x_i = get_stroke_sequence(stroke_fname)
        valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)

        c[i, :len(c_i)] = c_i
        c_len[i] = len(c_i)

        w_id[i] = w_id_i

    if not os.path.isdir(processed_data_path):
        os.makedirs(processed_data_path)

    np.save(f'{processed_data_path}/x.npy', x[valid_mask])
    np.save(f'{processed_data_path}/x_len.npy', x_len[valid_mask])
    np.save(f'{processed_data_path}/c.npy', c[valid_mask])
    np.save(f'{processed_data_path}/c_len.npy', c_len[valid_mask])
    np.save(f'{processed_data_path}/w_id.npy', w_id[valid_mask])
