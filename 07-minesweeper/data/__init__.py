#!/usr/bin/env python3

# %%
import numpy as np
import os.path

try:
    here = os.path.dirname(__file__)
except:
    here = os.path.dirname((os.getenv("HOME") or '.') + "/src/progml/07-minesweeper/dummy.py")

data_dir = str(here)


##
# minesweeper data:
# 111 mines, 97 rocks, labeled next to data of
# 60 x's -- eac of which is a number [0.0,1.0], rep'ing energy within a freq band over a period of
# time.
#
# small sample size means that random training divisions can lead to uneven results, as seen by
# original experimenters.
#

# %%
# lab
labels_ordered = ['R', 'M']
label_values = {l:i for i, l in enumerate(labels_ordered)}

# %%
def load_and_split_original():
    def default(f: str) -> float:
        return float(f)

    def map_labels(label: str) -> float:
        return label_values[label]

    # : dict[str, typing.Callable[[str], float]]
    column_converters = {i: default if i < 60 else map_labels for i in range(61)}

    Raw = np.loadtxt(
        data_dir + '/sonar.all-data',
        delimiter=',',
        # dtype=[(f'measure{i:02d}', np.float32) for i in range(60)] + [('label', str)],
        converters=column_converters,
        unpack=True
    )

    np.random.seed(1234)        # Have the same predictable shuffle every time
    np.random.shuffle(Raw.T)    # Shuffle matrix rows in place
    train_cnt = 160
    test_cnt = 208 - train_cnt
    assert Raw.shape[1] == train_cnt + test_cnt, f'dimension mismatch: {Raw.shape[1]} != {train_cnt} + {test_cnt}'
    print(f'Saving train={train_cnt} / test={test_cnt} split...')
    np.savetxt(data_dir + f'/sonar-all--train-{train_cnt}.data', Raw.T[:train_cnt])
    np.savetxt(data_dir + f'/sonar-all--test-{test_cnt}.data', Raw.T[train_cnt:])
    print(f'done.')

# %%
# load_and_split_original()

train = np.loadtxt(data_dir + f'/sonar-all--train-160.data', unpack=True).T
assert train.shape == (160, 61) # samples x (60 dims + 1 label)

test = np.loadtxt(data_dir + '/sonar-all--test-48.data', unpack=True).T
assert test.shape == (208 - 160, 61) # samples x (60 dims + 1 label)
# %%

