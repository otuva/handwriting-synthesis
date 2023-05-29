import numpy as np


def batch_generator(batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
    gen = df.batch_generator(
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs,
        allow_smaller_final_batch=(mode == 'test')
    )
    for batch in gen:
        batch['x_len'] = batch['x_len'] - 1
        max_x_len = np.max(batch['x_len'])
        max_c_len = np.max(batch['c_len'])
        batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
        batch['x'] = batch['x'][:, :max_x_len, :]
        batch['c'] = batch['c'][:, :max_c_len]
        yield batch
