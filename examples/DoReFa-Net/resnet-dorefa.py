#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: resnet-dorefa.py

import cv2
import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import get_imagenet_dataflow, eval_on_ILSVRC12, fbresnet_augmentor
from dorefa import get_dorefa

"""
This script loads the pre-trained ResNet-18 model with (W,A,G) = (1,4,32)
It has 59.2% top-1 and 81.5% top-5 validation error on ILSVRC12 validation set.

To run on images:
    ./resnet-dorefa.py --load pretrained.npy --run a.jpg b.jpg

To eval on ILSVRC validation set:
    ./resnet-dorefa.py --load pretrained.npy --eval --data /path/to/ILSVRC
"""

EPS = 16

BITW = 1
BITA = 4
BITG = 32

TOTAL_BATCH_SIZE = 32
BATCH_SIZE = None


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 256.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        def resblock(x, channel, stride):
            def get_stem_full(x):
                return (LinearWrap(x)
                        .Conv2D('c3x3a', channel, 3)
                        .BatchNorm('stembn')
                        .apply(activate)
                        .Conv2D('c3x3b', channel, 3)())
            channel_mismatch = channel != x.get_shape().as_list()[3]
            if stride != 1 or channel_mismatch or 'pool1' in x.name:
                # handling pool1 is to work around an architecture bug in our
                # model
                if stride != 1 or 'pool1' in x.name:
                    x = AvgPooling('pool', x, stride, stride)
                x = BatchNorm('bn', x)
                x = activate(x)
                shortcut = Conv2D('shortcut', x, channel, 1)
                stem = get_stem_full(x)
            else:
                shortcut = x
                x = BatchNorm('bn', x)
                x = activate(x)
                stem = get_stem_full(x)
            return shortcut + stem

        def group(x, name, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1'):
                x = resblock(x, channel, stride)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i)):
                    x = resblock(x, channel, 1)
            return x

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      # use explicit padding here, because our training framework has
                      # different padding mechanisms from TensorFlow
                      .tf.pad([[0, 0], [3, 2], [3, 2], [0, 0]])
                      .Conv2D('conv1', 64, 7, stride=2, padding='VALID', use_bias=True)
                      .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
                      .MaxPooling('pool1', 3, 2, padding='VALID')
                      .apply(group, 'conv2', 64, 2, 1)
                      .apply(group, 'conv3', 128, 2, 2)
                      .apply(group, 'conv4', 256, 2, 2)
                      .apply(group, 'conv5', 512, 2, 2)
                      .BatchNorm('lastbn')
                      .apply(nonlin)
                      .GlobalAvgPooling('gap')
                      # this is due to a bug in our model design
                      #.tf.multiply(49)
                      .FullyConnected('fct', 1000)())
        tf.nn.softmax(logits, name='output')

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

        # weight decay on all W of fc layers
        wd_cost_1 = regularize_cost(
            'fc.*/W', l2_regularizer(1e-4), name='regularize_cost')

        # weight decay on conv0 fp conv layer
        wd_cost_2 = regularize_cost(
            'conv0/W', l2_regularizer(1e-5), name='regularize_cost')

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost_1, wd_cost_2], name='cost')
        add_moving_summary(cost, wd_cost_1, self.cost)

        #ImageNetModel.compute_loss_and_error(logits, image, label, EPS)
        # eps = 16.0 # maximum size of adversarial perturbation
        #ImageNetModel.compute_loss_and_error(logits, image, label, eps)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, dataset_name, BATCH_SIZE, augmentors)


def get_config(name):
    # when running under job scheduler, always create new
    logger.auto_set_dir(action='n', name=name)
    data_train = get_data('train')
    data_test = get_data('val')

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError(
                                 'wrong-top1', 'val-error-top1'),
                             ClassificationError('wrong-top5', 'val-error-top5')])
        ],
        model=Model(),
        steps_per_epoch=5000,
        max_epoch=110,
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
    predict_func = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    words = meta.get_synset_words_1000()

    transformers = get_inference_augmentor()
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        o = predict_func(img)
        prob = o[0][0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a npy pretrained model')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument(
        '--dorefa', help='number of bits for W,A,G, separated by comma')
    parser.add_argument(
        '--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--attack', action='store_true')
    parser.add_argument("--eps", type=float, default=16.0)
    args = parser.parse_args()

    if args.dorefa:
        BITW, BITA, BITG = map(int, args.dorefa.split(','))
    dorefa_string = str(BITW) + '-' + str(BITA) + '-' + str(BITG) + '__'
    EPS = args.eps

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.eval:
        from imagenet_utils import eval_on_ILSVRC12
        ds = dataset.ILSVRC12(args.data, 'val', shuffle=False)
        ds = AugmentImageComponent(ds, get_inference_augmentor())
        ds = BatchData(ds, 384, remainder=True)
        eval_on_ILSVRC12(Model(), get_model_loader(args.load), ds)

    elif args.attack:
        from imagenet_utils import attack_on_ILSVRC12
        ds = dataset.ILSVRC12(args.data, 'val', shuffle=False)
        ds = AugmentImageComponent(ds, get_inference_augmentor())
        ds = BatchData(ds, 192, remainder=True)
        print("Attacking with FGSM eps = %.2f" % EPS)
        attack_on_ILSVRC12(Model(), get_model_loader(args.load), ds)

    elif args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), DictRestore(
            np.load(args.load, encoding='latin1').item()), args.run)

    else:
        nr_tower = max(get_nr_gpu(), 1)
        BATCH_SIZE = TOTAL_BATCH_SIZE // nr_tower
        logger.info("Batch per tower: {}".format(BATCH_SIZE))

        config = get_config(dorefa_string)
        if args.load:
            if args.load.endswith('.npy'):
                config.session_init = get_model_loader(args.load)
            else:
                config.session_init = SaverRestore(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))
