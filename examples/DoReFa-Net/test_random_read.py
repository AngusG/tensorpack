import tensorpack
from tensorpack.dataflow import *
ds0 = dataset.ILSVRC12('/scratch/gallowaa/imagenet/imagenet-raw/raw-data', 'train', shuffle=True)
ds1 = BatchData(ds0, 256, use_list=True)
TestDataSpeed(ds1).start()
