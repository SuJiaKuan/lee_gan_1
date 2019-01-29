import numpy as np


class DataSet(object):

    def __init__(self, data):
        self._data = np.array(data)
        self._data_size = self._data.shape[0]

    def sample(self, batch_size):
        idx_list = np.random.choice(self._data_size,
                                    size=batch_size,
                                    replace=False)

        return self._data[idx_list]
