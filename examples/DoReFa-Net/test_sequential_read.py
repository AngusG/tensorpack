'''
# 3.5 it/s
import tensorpack
from tensorpack.dataflow import *
ds = LMDBData('/scratch/gallowaa/imagenet/ILSVRC12-train.lmdb', shuffle=False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start()
'''

'''
# 0.9 it/s
import cv2
import tensorpack
from tensorpack.dataflow import *
ds = LMDBData('/scratch/gallowaa/imagenet/ILSVRC12-train.lmdb', shuffle=False)
ds = LocallyShuffleData(ds, 50000)
ds = LMDBDataPoint(ds)
ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
augmentors = [
    imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
    imgaug.CenterCrop((224,224)),
]
ds = AugmentImageComponent(ds, augmentors)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start()
'''

'''
Launch base DataFlow in one process, and parallelize the augmentation with
PrefetchDataZMQ.
'''
# 2.9 it/s
import cv2
import tensorpack
from tensorpack.dataflow import *
ds = LMDBData('/scratch/gallowaa/imagenet/ILSVRC12-train.lmdb', shuffle=False)
ds = LocallyShuffleData(ds, 50000)
ds = PrefetchData(ds, 5000, 1)
ds = LMDBDataPoint(ds)
ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
augmentors = [
    imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
    imgaug.CenterCrop((224,224)),
]
ds = AugmentImageComponent(ds, augmentors, copy=False)
ds = PrefetchDataZMQ(ds, 25) # nproc + 1
ds = BatchData(ds, 256, remainder=False)
TestDataSpeed(ds).start()

