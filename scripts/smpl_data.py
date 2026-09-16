"""Restricted reader for numeric joblib SMPL data; never imports pickle globals."""
import pickle
from pathlib import Path
import numpy as np


class ArrayWrapper:
    pass


class NumericUnpickler(pickle._Unpickler):
    def find_class(self, module, name):
        allowed = {('joblib.numpy_pickle', 'NumpyArrayWrapper'): ArrayWrapper,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported serialized global: {module}.{name}')
        return allowed[module, name]

    def load_build(self):
        super().load_build()
        wrapper = self.stack[-1]
        if not isinstance(wrapper, ArrayWrapper):
            return
        dtype = wrapper.dtype
        if dtype.hasobject or dtype.kind not in 'biufc':
            raise ValueError(f'Only numeric arrays are supported: {dtype}')
        count = int(np.prod(wrapper.shape))
        if count < 0 or count * dtype.itemsize > 1024**3:
            raise ValueError('Invalid array size')
        if getattr(wrapper, 'numpy_array_alignment_bytes', None):
            padding = self.read(1)[0]
            self.read(padding)
        raw = self.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise ValueError('Truncated array')
        self.stack[-1] = np.frombuffer(raw, dtype=dtype).reshape(wrapper.shape, order=wrapper.order).copy()

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.BUILD[0]] = load_build


def load_numeric(path):
    with Path(path).open('rb') as stream:
        return NumericUnpickler(stream).load()


if __name__ == '__main__':
    import sys
    def describe(x, prefix=''):
        if isinstance(x, dict):
            for k, v in x.items():
                describe(v, prefix + '/' + str(k))
        else:
            print(prefix, type(x).__name__, getattr(x, 'shape', ''),
                  (str(x)[:100] if not isinstance(x, np.ndarray) else str(x.dtype)))
    describe(load_numeric(sys.argv[1]))
