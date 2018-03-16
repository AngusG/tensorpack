#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: alexnet-dorefa.py
# Author: Yuxin Wu, Yuheng Zou ({wyx,zyh}@megvii.com)

import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import sys


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

from misc_utils import contains_lmdb
from imagenet_utils import get_imagenet_dataflow, eval_on_ILSVRC12, fbresnet_augmentor
from dorefa import get_dorefa

"""
This is a tensorpack script for the ImageNet results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack & TensorFlow.

Accuracy:
    Trained with 4 GPUs and (W,A,G)=(1,2,6), it can reach top-1 single-crop validation error of 47.6%,
    after 70 epochs. This number is better than what's in the paper
    due to more sophisticated augmentations.

    With (W,A,G)=(32,32,32) -- full precision baseline, 41.4% error.
    With (W,A,G)=(1,32,32) -- BWN, 44.3% error
    With (W,A,G)=(1,2,6), 47.6% error
    With (W,A,G)=(1,2,4), 58.4% error

Speed:
    About 11 iteration/s on 4 P100s. (Each epoch is set to 10000 iterations)
    Note that this code was written early without using NCHW format. You
    should expect a speed up if the code is ported to NCHW format.

To Train, for example:
    ./alexnet-dorefa.py --dorefa 1,2,6 --data PATH --gpu 0,1

    PATH should look like:
    PATH/
      train/
        n02134418/
          n02134418_198.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

    And you'll need the following to be able to fetch data efficiently
        Fast disk random access (Not necessarily SSD. I used a RAID of HDD, but not sure if plain HDD is enough)
        More than 20 CPU cores (for data processing)
        More than 10G of free memory

To Run Pretrained Model:
    ./alexnet-dorefa.py --load alexnet-126.npy --run a.jpg --dorefa 1,2,6
"""

EPS = 16

BITW = 1
BITA = 2
BITG = 6

L2_DECAY = 1e-4

TOTAL_BATCH_SIZE = 128
BATCH_SIZE = None

FCT = 'fct/W'
CONV0 = 'conv0/W'
EXCLUDE = [CONV0, FCT]


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            # if not name.endswith('W') or 'conv0' in name or 'fct' in name:
            if not name.endswith('W') or name in EXCLUDE:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 96, 12, stride=8, padding='SAME')
                      .apply(activate)
                      #.Conv2D('conv1', 256, 5, padding='SAME', split=2)
                      .apply(fg)
                      #.BatchNorm('bn1')
                      #.MaxPooling('pool1', 3, 2, padding='SAME')
                      .apply(activate)

                      #.Conv2D('conv2', 384, 3)
                      .Conv2D('conv1', 384, 6, stride=3)
                      .apply(fg)
                      #.BatchNorm('bn2')
                      #.MaxPooling('pool2', 3, 2, padding='SAME')
                      .apply(activate)

                      #.Conv2D('conv3', 384, 3, split=2)
                      .Conv2D('conv2', 256, 3, stride=3)
                      .apply(fg)
                      #.BatchNorm('bn3')
                      .apply(activate)

                      #.Conv2D('conv4', 256, 3, split=2)
                      #.Conv2D('conv4', 256, 3, stride=2)
                      .apply(fg)
                      #.BatchNorm('bn4')
                      #.MaxPooling('pool4', 3, 2, padding='VALID')
                      .apply(activate)

                      #.FullyConnected('fc0', 2048)
                      #.apply(fg)
                      #.BatchNorm('bnfc0')
                      #.apply(activate)

                      #.FullyConnected('fc1', 4096)
                      .apply(fg)
                      #.BatchNorm('bnfc1')
                      .apply(nonlin)
                      .FullyConnected('fct', 1000, use_bias=True)())

        output = tf.nn.softmax(logits, name='output')
        #correct = tf.equal(tf.argmax(output, 1), label)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        grads = tf.gradients(cost, image)[0]
        adv_image = tf.clip_by_value(
            image + EPS / 255.0 * tf.sign(grads), 0, 1, name='adv_x')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        loss_terms = [cost]

        # weight decay on all W of fc layers
        # if FCT in EXCLUDE:
        wd_cost_fct = regularize_cost(
            'fc.*/W', l2_regularizer(L2_DECAY), name='regularize_fc')
        loss_terms.append(wd_cost_fct)

        # weight decay on conv0 fp conv layer
        # if CONV0 in EXCLUDE:
        wd_cost_conv0 = regularize_cost(
            'conv*/W', l2_regularizer(L2_DECAY), name='regularize_conv')
            #CONV0, l2_regularizer(L2_DECAY), name='regularize_conv0')
        loss_terms.append(wd_cost_conv0)

        add_param_summary(('.*/W', ['histogram', 'rms']))
        #self.cost = tf.add_n([cost, wd_cost_1, wd_cost_2], name='cost')
        self.cost = tf.add_n(loss_terms, name='cost')
        add_moving_summary(cost, wd_cost_fct, self.cost)

    def _get_optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=1e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_data(dataset_name, applyCutout=False):
    isTrain = dataset_name == 'train'
    augmentors = fbresnet_augmentor(isTrain, applyCutout)
    return get_imagenet_dataflow(
        args.data, dataset_name, BATCH_SIZE, augmentors)


def get_config(name, applyCutout):
    # when running under job scheduler, always create new
    logger.auto_set_dir(action='n', name=name)
    data_train = get_data('train', applyCutout)
    data_test = get_data('val')

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(56, 2e-5), (64, 4e-6)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError(
                                 'wrong-top1', 'val-error-top1'),
                             ClassificationError('wrong-top5', 'val-error-top5')])
        ],
        model=Model(),
        steps_per_epoch=10000,
        max_epoch=100,
    )


def get_inference_augmentor():
    return fbresnet_augmentor(False)


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predictor = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        outputs = predictor(img)[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument(
        '--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma')
    parser.add_argument(
        '--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument(
        '--eval', help='evaluate model on fgsm if --eps provided, otherwise clean', action='store_true')
    parser.add_argument(
        "--l2", help='L2 regularization applied to non-quantized layers', type=float, default=1e-4)
    parser.add_argument(
        "--eps", help='magnitude of perturbation, use with --eval', type=float)
    parser.add_argument("--ps", help='location of parameter server',
                        default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument(
        '--first', help='quantize first layer', action='store_true')
    parser.add_argument(
        '--cutout', help='apply cutout', action='store_true')
    args = parser.parse_args()

    if args.dorefa:
        BITW, BITA, BITG = map(int, args.dorefa.split(','))
    L2_DECAY = args.l2
    model_details = str(BITW) + '-' + str(BITA) + '-' + \
        str(BITG) + "_{0:.1e}".format(L2_DECAY) + '_l2_'
    if args.first:
        EXCLUDE = [FCT]
        model_details += 'conv0_'

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.eval:
        # eval from lmdb (usually faster)
        if contains_lmdb(args.data):
            BATCH_SIZE = 128
            logger.info("Batch per tower: {}".format(BATCH_SIZE))
            ds = get_data('val')
        # eval from raw images (very slow unless data on ssd)
        else:
            ds = dataset.ILSVRC12(args.data, 'val', shuffle=False)
            ds = AugmentImageComponent(ds, get_inference_augmentor())
            ds = BatchData(ds, 128, remainder=True)

        # attack with fgsm if epsilon provided
        if args.eps:
            from imagenet_utils import attack_on_ILSVRC12
            EPS = args.eps
            print("Attacking with FGSM eps = %.2f" % EPS)
            attack_on_ILSVRC12(Model(), get_model_loader(
                args.load), ds)
        else:
            from imagenet_utils import eval_on_ILSVRC12
            print("Evaluating on clean data")
            eval_on_ILSVRC12(Model(), get_model_loader(args.load), ds)

    elif args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), DictRestore(
            np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    else:
        nr_tower = max(get_nr_gpu(), 1)
        BATCH_SIZE = TOTAL_BATCH_SIZE // nr_tower
        logger.info("Batch per tower: {}".format(BATCH_SIZE))

        applyCutout = False
        if args.cutout:
            model_details += 'cut_'
            applyCutout = True
        config = get_config(model_details, applyCutout)
        if args.load:
            if args.load.endswith('.npy'):
                config.session_init = get_model_loader(args.load)
            else:
                config.session_init = SaverRestore(args.load)
        trainer = SyncMultiGPUTrainerParameterServer(
            nr_tower, ps_device=args.ps)
        launch_train_with_config(config, trainer)
