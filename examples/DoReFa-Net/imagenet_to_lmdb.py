import numpy as np
base_path = '/scratch/gallowaa/imagenet/'
raw_path = 'imagenet-raw/raw-data'

from tensorpack.dataflow import *
class BinaryILSVRC12(dataset.ILSVRC12Files):
    def get_data(self):
        for fname, label in super(BinaryILSVRC12, self).get_data():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]

ds0 = BinaryILSVRC12(base_path + raw_path, 'val')
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, base_path + 'ILSVRC12-val.lmdb')
