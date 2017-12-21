#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """

    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(
                self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(
                    out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(
            self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from
                 # fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )

    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


def attack_on_ILSVRC12(model, sessinit, dataflow):

    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['adv_x', 'wrong-top1', 'wrong-top5']
    )

    # x = pred.predictor.input_tensors[0]  # (?, 224,224,3)
    # loss = pred.predictor.output_tensors[0] # (?, 1000) loss
    # evaluate a batch of adversarial images
    #for adv_x in pred.get_result():
        # x is the data (batch_size, 224, 224, 3) , y is the label (batch_size)
    #for _, y in pred.dataset.get_data():
        #sess.run(wrong_top1, feed_dict={x: adv_x, label: y})

    pred = SimpleDatasetPredictor(pred_config, dataflow)

    sess = pred.predictor.sess
    x = pred.predictor.input_tensors[0]
    label = pred.predictor.input_tensors[1]
    wrong_top1 = pred.predictor.output_tensors[1]
    wrong_top5 = pred.predictor.output_tensors[2]

    batch_size = 192
    cln_acc1, cln_acc5 = RatioCounter(), RatioCounter()
    adv_acc1, adv_acc5 = RatioCounter(), RatioCounter()

    for ((adv_x, cln_top1, cln_top5), (__, y)) in zip(pred.get_result(), pred.dataset.get_data()):
        
        adv_top1 = sess.run(wrong_top1, feed_dict={x: adv_x, label: y})        
        adv_top5 = sess.run(wrong_top5, feed_dict={x: adv_x, label: y})        
        
        adv_acc1.feed(adv_top1.sum(), batch_size)
        adv_acc5.feed(adv_top5.sum(), batch_size)
        cln_acc1.feed(cln_top1.sum(), batch_size)
        cln_acc5.feed(cln_top5.sum(), batch_size)

        print("Cln Top1 Error: {}".format(cln_acc1.ratio))
        print("Cln Top5 Error: {}".format(cln_acc5.ratio))
        print("Adv Top1 Error: {}".format(adv_acc1.ratio))
        print("Adv Top5 Error: {}".format(adv_acc5.ratio))

    '''
    acc1, acc5 = RatioCounter(), RatioCounter()    
    for adv_x, top1, top5 in pred.get_result():
        batch_size = top1.shape[0]        
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))
    '''


class ImageNetModel(ModelDesc):
    weight_decay = 1e-4
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        print("Hello from _build_graph()")
        image, label = inputs
        image = self.image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(logits, label)
        wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        self.cost = tf.add_n([loss, wd_loss], name='cost')

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Nx1000 logits
        """

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, image, label, eps):
        print("Hello from compute_loss_and_error()")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        grads = tf.gradients(loss, image)[0]
        adv_image = tf.clip_by_value(
            image + eps / 256.0 * tf.sign(grads), 0, 1, name='adv_x')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss
