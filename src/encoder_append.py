import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from keras.layers import Input,Conv2D,Add,BatchNormalization,Activation, Lambda, Multiply, Conv2DTranspose, Concatenate, ZeroPadding2D
from keras.layers.convolutional import AtrousConv2D
from keras.models import Model
from keras.utils import plot_model,np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler, \
    TensorBoard,ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.regularizers import l2
from loss import detection_focal_loss_K, detection_loss_K, detection_double_focal_loss_K, detection_double_focal_loss_indicator_K
from config import Config
from tensorflow.python.client import device_lib
#from encoder_decoder_object_det import Conv3l2
from util import *
from encoder_decoder_object_det import callback_preparation, crop_shape_generator_with_heavy_aug, TimerCallback
import scipy.misc as misc
from eval import mymetrics
from loss import cls_cross_entropy

weight_decay = 0.005
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')
#CROP_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')


def _image_normalization(image, preprocss_num):
    """
    preprocessing on image.
    """
    image = image - preprocss_num
    image = image / preprocss_num
    return image


def load_dataset(dataset, type, reshape_size=None, det=True, cls=True, preprocss_num=128.):
    """
    Load dataset from files
    :param type: either train, test or validation.
    :param reshape_size: reshape to (512, 512) if cropping images are using.
    :param det: True if detection masks needed.
    :param cls: True if classification masks needed.
    :param preprocss_num: number to subtract and divide in normalization step.
    """
    path = os.path.join(dataset, type)
    imgs, det_masks, cls_masks = [], [], []
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            if 'original.bmp' in img_file:
                img_path = os.path.join(path, file, img_file)
                img = misc.imread(img_path)
                if reshape_size is not None:
                    img = misc.imresize(img, reshape_size, interp='nearest')
                img = _image_normalization(img, preprocss_num)
                imgs.append(img)
            if 'detection.bmp' in img_file and det is True and 'verifiy' not in img_file:
                det_mask_path = os.path.join(path, file, img_file)
                det_mask = misc.imread(det_mask_path, mode='L')
                if reshape_size is not None:
                    det_mask = misc.imresize(det_mask, reshape_size, interp='nearest')
                det_mask = det_mask.reshape(det_mask.shape[0], det_mask.shape[1], 1)
                det_masks.append(det_mask)

            if 'classification.bmp' in img_file and cls is True and 'verifiy' not in img_file:
                cls_mask_path = os.path.join(path, file, img_file)
                cls_mask = misc.imread(cls_mask_path, mode='L')
                if reshape_size != None:
                    cls_mask = misc.imresize(cls_mask, reshape_size, interp='nearest')
                #cls_mask = cls_mask.reshape(cls_mask.shape[0], cls_mask.shape[1], 1)
                cls_masks.append(cls_mask)

    print(len(imgs), len(det_masks), len(cls_masks))
    #imgs = _torch_image_transpose(imgs)
    #det_masks = _torch_image_transpose(det_masks)
    #cls_masks = _torch_image_transpose(cls_masks)
    return np.array(imgs), np.array(det_masks), np.array(cls_masks)


class Conv3l2(keras.layers.Conv2D):
    """
    Custom convolution layer with default 3*3 kernel size and L2 regularization.
    Default padding change to 'same' in this case.
    """
    def __init__(self, filters, kernel_regularizer_weight,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(self.__class__, self).__init__(filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding=padding,
                                             data_format=data_format,
                                             dilation_rate=dilation_rate,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=l2(kernel_regularizer_weight),
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             kernel_constraint=kernel_constraint,
                                             bias_constraint=bias_constraint,
                                             **kwargs)
###########################################
# ResNet Graph
###########################################
def identity_block_2(f, stage, block, inputs, l2_weight, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs

    x = Conv2D(filters=f, kernel_size=(3,3), padding='same',
                name=str(stage) + '_' + str(block) + '_idblock_conv_1',
                trainable=trainable)(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_1',
                           trainable=trainable)(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_1',
                   trainable=trainable)(x)

    x = Conv2D(filters=f, kernel_size=(3,3), padding='same',
                name=str(stage) + '_' + str(block) + '_idblock_conv_2')(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_2',
                           trainable=trainable)(x)

    x_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add', trainable=trainable)([x, x_shortcut])
    x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_outout',
                                  trainable=trainable)(x_add)
    return x_idblock_output


def convolution_block_2(f, stage, block, inputs, l2_weight, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    """
    x = Conv2D(filters=f, strides=(2, 2), kernel_size=(3,3), padding='same',
                name=str(stage) + str(block) + '_' + '_convblock_conv_1',
                trainable=trainable)(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_BN_1',
                           trainable=trainable)(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_1',
                   trainable=trainable)(x)
    x = Conv2D(filters=f, kernel_size=(3,3), padding='same',
                name=str(stage) + '_' + str(block) + '_convblock_conv_2',
                trainable=trainable)(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_BN_2',
                           trainable=trainable)(x)


    x_shortcut = Conv2D(f, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                        trainable=trainable)(inputs)
    x_shortcut = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_shortcut_BN_1',
                                    trainable=trainable)(x_shortcut)
    x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add',
                trainable=trainable)([x, x_shortcut])
    x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_output',
                                    trainable=trainable)(x_add)
    return x_convblock_output

def dilated_bottleneck(inputs, stage, block):
    """
    Dilated block without 1x1 convolution projection, structure like res-id-block
    """
    x_shortcut = inputs
    x = Conv2D(filters=256, kernel_size=(1,1), padding='same',
               name=str(stage) + '_' + str(block) + '_1' + '_dilated_block_first1x1')(inputs)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstBN')(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_1'+ '_dilated_block_firstRELU')(x)

    x_dilated = Conv2D(filters=256, kernel_size=(3,3), padding='same',
                       name=str(stage) + '_' + str(block) + '_2' + '_dilated_block_dilatedconv', dilation_rate=(2,2))(x)
    x_dilated = BatchNormalization(name=str(stage) + '_'+ str(block) + '_2'+ '_dilated_block_dilatedBN')(x_dilated)
    x_dilated = Activation('relu',name=str(stage) +'_'+ str(block) + '_2'+ '_dilated_block_dilatedRELU')(x_dilated)

    x_more = Conv2D(filters=256, kernel_size=(1,1), padding='same',
                    name=str(stage) + '_'+ str(block) + '_3'+ '_dilated_block_second1x1')(x_dilated)
    x_more = BatchNormalization(name=str(stage) + '_' + str(block) + '_3'+ '_dilated_block_secondBN')(x_more)
    x_more = Activation('relu', name=str(stage) + str(block) + '_3' +'_dilated_block_secondRELU')(x_more)
    x_add = Add(name=str(stage) + '_' + str(block) + '_3' + '_dilated_block_Add')([x_more, x_shortcut])
    x_dilated_output = Activation('relu', name=str(stage)+'_' + str(block) +'_dilated_block_relu')(x_add)
    return x_dilated_output

def dilated_with_projection(inputs, stage):
    """
    Dilated block with 1x1 convolution projection for the shortcut, structure like res-conv-block
    """
    x_shortcut = inputs
    x = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
               name=str(stage) + '_1'+ '_dilated_project_first1x1')(inputs)
    x = BatchNormalization(name=str(stage) + '_1'+ '_dilated_project_firstBN')(x)
    x = Activation('relu', name=str(stage) + '_1''_dilated_project_firstRELU')(x)

    x_dilated = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                       name=str(stage) + '_2'+'_dilated_project_dilatedconv', dilation_rate=(2, 2))(x)
    x_dilated = BatchNormalization(name=str(stage)+ '_2' + '_dilated_project_DBN')(x_dilated)
    x_dilated = Activation('relu', name=str(stage) + '_2'+ '_dialated_project_DRELU')(x_dilated)

    x_more = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                    name=str(stage) + '_3'+ '_dilated_project_second1x1')(x_dilated)
    x_more = BatchNormalization(name=str(stage) + '_3'+ '_dilated_project_secondBN')(x_more)
    x_more = Activation('relu',name=str(stage) + '_3'+ '_dilated_project_secondRELU')(x_more)

    x_shortcut_project = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
               name=str(stage) + '_dialted_project_shortcutConv')(x_shortcut)
    x_shortcut_project = BatchNormalization(name=str(stage) + '_dialted_project_shortcutBN')(x_shortcut_project)

    x_add = Add(name=str(stage) + '_dilated_project_add')([x_more, x_shortcut_project])
    x_dilated_output = Activation('relu', name=str(stage) + '_dilated_project_finalRELU')(x_add)
    return x_dilated_output


def fcn_27(inputs, l2_weight, stages=[2, 3, 4, 5], filters=[32, 64, 128], trainable=True):
    x = inputs
    x_id_20 = identity_block_2(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable, l2_weight=l2_weight)
    x_id_21 = identity_block_2(f=filters[0], stage=stages[0], block=2, inputs=x_id_20, trainable=trainable, l2_weight=l2_weight)
    x_id_22 = identity_block_2(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable, l2_weight=l2_weight)
    x_id_23 = identity_block_2(f=filters[0], stage=stages[0], block=4, inputs=x_id_22, trainable=trainable, l2_weight=l2_weight)
    x_id_24 = identity_block_2(f=filters[0], stage=stages[0], block=5, inputs=x_id_23, trainable=trainable, l2_weight=l2_weight)
    x_id_25 = identity_block_2(f=filters[0], stage=stages[0], block=6, inputs=x_id_24, trainable=trainable, l2_weight=l2_weight)
    x_id_26 = identity_block_2(f=filters[0], stage=stages[0], block=7, inputs=x_id_25, trainable=trainable, l2_weight=l2_weight)
    x_id_27 = identity_block_2(f=filters[0], stage=stages[0], block=8, inputs=x_id_26, trainable=trainable, l2_weight=l2_weight)
    x_id_28 = identity_block_2(f=filters[0], stage=stages[0], block=9, inputs=x_id_27, trainable=trainable, l2_weight=l2_weight)

    x_conv_3 = convolution_block_2(f=filters[1], stage=stages[1], block=1, inputs=x_id_28, trainable=trainable, l2_weight=l2_weight)
    x_id_31 = identity_block_2(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable, l2_weight=l2_weight)
    x_id_32 = identity_block_2(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable, l2_weight=l2_weight)
    x_id_33 = identity_block_2(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable, l2_weight=l2_weight)
    x_id_34 = identity_block_2(f=filters[1], stage=stages[1], block=5, inputs=x_id_33, trainable=trainable, l2_weight=l2_weight)
    x_id_35 = identity_block_2(f=filters[1], stage=stages[1], block=6, inputs=x_id_34, trainable=trainable, l2_weight=l2_weight)
    x_id_36 = identity_block_2(f=filters[1], stage=stages[1], block=7, inputs=x_id_35, trainable=trainable, l2_weight=l2_weight)
    x_id_37 = identity_block_2(f=filters[1], stage=stages[1], block=8, inputs=x_id_36, trainable=trainable, l2_weight=l2_weight)
    x_id_38 = identity_block_2(f=filters[1], stage=stages[1], block=9, inputs=x_id_37, trainable=trainable, l2_weight=l2_weight)


    x_conv_4 = convolution_block_2(f=filters[2], stage=stages[2], block=1, inputs=x_id_38, trainable=trainable, l2_weight=l2_weight)
    x_id_41 = identity_block_2(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable, l2_weight=l2_weight)
    x_id_42 = identity_block_2(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable, l2_weight=l2_weight)
    x_id_43 = identity_block_2(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable, l2_weight=l2_weight)
    x_id_44 = identity_block_2(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable, l2_weight=l2_weight)
    x_id_45 = identity_block_2(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable, l2_weight=l2_weight)
    x_id_46 = identity_block_2(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable, l2_weight=l2_weight)
    x_id_47 = identity_block_2(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable, l2_weight=l2_weight)
    x_id_48 = identity_block_2(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable, l2_weight=l2_weight)
    return x_id_28, x_id_38, x_id_48

def fcn_36(inputs, l2_weight, stages=[2, 3, 4, 5],filters=[32, 64, 128, 256], trainable=True):
    x = inputs
    x_id_20 = identity_block_2(f=filters[0], stage=stages[0], block=1, inputs=x, trainable=trainable, l2_weight=l2_weight)
    x_id_21 = identity_block_2(f=filters[0], stage=stages[0], block=2, inputs=x_id_20, trainable=trainable, l2_weight=l2_weight)
    x_id_22 = identity_block_2(f=filters[0], stage=stages[0], block=3, inputs=x_id_21, trainable=trainable, l2_weight=l2_weight)
    x_id_23 = identity_block_2(f=filters[0], stage=stages[0], block=4, inputs=x_id_22, trainable=trainable, l2_weight=l2_weight)
    x_id_24 = identity_block_2(f=filters[0], stage=stages[0], block=5, inputs=x_id_23, trainable=trainable, l2_weight=l2_weight)
    x_id_25 = identity_block_2(f=filters[0], stage=stages[0], block=6, inputs=x_id_24, trainable=trainable, l2_weight=l2_weight)
    x_id_26 = identity_block_2(f=filters[0], stage=stages[0], block=7, inputs=x_id_25, trainable=trainable, l2_weight=l2_weight)
    x_id_27 = identity_block_2(f=filters[0], stage=stages[0], block=8, inputs=x_id_26, trainable=trainable, l2_weight=l2_weight)
    x_id_28 = identity_block_2(f=filters[0], stage=stages[0], block=9, inputs=x_id_27, trainable=trainable, l2_weight=l2_weight)

    x_conv_3 = convolution_block_2(f=filters[1], stage=stages[1], block=1, inputs=x_id_28, trainable=trainable, l2_weight=l2_weight)
    x_id_31 = identity_block_2(f=filters[1], stage=stages[1], block=2, inputs=x_conv_3, trainable=trainable, l2_weight=l2_weight)
    x_id_32 = identity_block_2(f=filters[1], stage=stages[1], block=3, inputs=x_id_31, trainable=trainable, l2_weight=l2_weight)
    x_id_33 = identity_block_2(f=filters[1], stage=stages[1], block=4, inputs=x_id_32, trainable=trainable, l2_weight=l2_weight)
    x_id_34 = identity_block_2(f=filters[1], stage=stages[1], block=5, inputs=x_id_33, trainable=trainable, l2_weight=l2_weight)
    x_id_35 = identity_block_2(f=filters[1], stage=stages[1], block=6, inputs=x_id_34, trainable=trainable, l2_weight=l2_weight)
    x_id_36 = identity_block_2(f=filters[1], stage=stages[1], block=7, inputs=x_id_35, trainable=trainable, l2_weight=l2_weight)
    x_id_37 = identity_block_2(f=filters[1], stage=stages[1], block=8, inputs=x_id_36, trainable=trainable, l2_weight=l2_weight)
    x_id_38 = identity_block_2(f=filters[1], stage=stages[1], block=9, inputs=x_id_37, trainable=trainable, l2_weight=l2_weight)


    x_conv_4 = convolution_block_2(f=filters[2], stage=stages[2], block=1, inputs=x_id_38, trainable=trainable, l2_weight=l2_weight)
    x_id_41 = identity_block_2(f=filters[2], stage=stages[2], block=2, inputs=x_conv_4, trainable=trainable, l2_weight=l2_weight)
    x_id_42 = identity_block_2(f=filters[2], stage=stages[2], block=3, inputs=x_id_41, trainable=trainable, l2_weight=l2_weight)
    x_id_43 = identity_block_2(f=filters[2], stage=stages[2], block=4, inputs=x_id_42, trainable=trainable, l2_weight=l2_weight)
    x_id_44 = identity_block_2(f=filters[2], stage=stages[2], block=5, inputs=x_id_43, trainable=trainable, l2_weight=l2_weight)
    x_id_45 = identity_block_2(f=filters[2], stage=stages[2], block=6, inputs=x_id_44, trainable=trainable, l2_weight=l2_weight)
    x_id_46 = identity_block_2(f=filters[2], stage=stages[2], block=7, inputs=x_id_45, trainable=trainable, l2_weight=l2_weight)
    x_id_47 = identity_block_2(f=filters[2], stage=stages[2], block=8, inputs=x_id_46, trainable=trainable, l2_weight=l2_weight)
    x_id_48 = identity_block_2(f=filters[2], stage=stages[2], block=9, inputs=x_id_47, trainable=trainable, l2_weight=l2_weight)

    x_conv_5 = convolution_block_2(f=filters[3], stage=stages[3], block=1, inputs=x_id_48, trainable=trainable,
                                   l2_weight=l2_weight)
    x_id_51 = identity_block_2(f=filters[3], stage=stages[3], block=2, inputs=x_conv_5, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_52 = identity_block_2(f=filters[3], stage=stages[3], block=3, inputs=x_id_51, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_53 = identity_block_2(f=filters[3], stage=stages[3], block=4, inputs=x_id_52, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_54 = identity_block_2(f=filters[3], stage=stages[3], block=5, inputs=x_id_53, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_55 = identity_block_2(f=filters[3], stage=stages[3], block=6, inputs=x_id_54, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_56 = identity_block_2(f=filters[3], stage=stages[3], block=7, inputs=x_id_55, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_57 = identity_block_2(f=filters[3], stage=stages[3], block=8, inputs=x_id_56, trainable=trainable,
                               l2_weight=l2_weight)
    x_id_58 = identity_block_2(f=filters[3], stage=stages[3], block=9, inputs=x_id_57, trainable=trainable,
                               l2_weight=l2_weight)
    return x_id_28, x_id_38, x_id_48, x_id_58

class Fcn_det:
    def __init__(self, input_shape=(256, 256, 3)):
        # self.inputs = inputs
        self.input_shape = input_shape
        l2_weight = 0.001

    def first_layer(self, inputs, l2_weight, trainable=True):
        """
        First convolution layer.
        """
        x = Conv3l2(filters=32, name='Conv_1',
                    kernel_regularizer_weight=l2_weight,
                    trainable=trainable)(inputs)
        x = BatchNormalization(name='BN_1',trainable=trainable)(x)
        x = Activation('relu', name='act_1',trainable=trainable)(x)
        return x

    ####################################
    # FCN 36 as in paper sfcn-opi
    ####################################
    def fcn36_deconv_backbone(self, l2_weight=0.001):
        #######################################
        # Extra branch for every pyramid feature
        #######################################
        def _feature_concat_deconv_branch(C7=None, C6=None, C5=None, C4=None, C3=None):
            """

            :param features: input feature is from every feature pyramid layer,
                             should've been already connect with 1x1 convolution layer.
            """

            def _32to256(input, type):
                x_deconv64 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_64_Conv',
                                             padding='same',
                                             kernel_regularizer=l2(l2_weight))(input)
                x_deconv64 = BatchNormalization(name=type + '_deconv_64_BN')(x_deconv64)
                x_deconv64 = Activation('relu', name=type + '_deconv_64_RELU')(x_deconv64)
                x_deconv128 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_128_Conv',
                                              padding='same',
                                             kernel_regularizer=l2(l2_weight))(x_deconv64)
                x_deconv128 = BatchNormalization(name=type + '_deconv_128_BN')(x_deconv128)
                x_deconv128 = Activation('relu', name=type + '_deconv_128_RELU')(x_deconv128)
                x_deconv256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                              filters=256, strides=(2, 2), name=type + '_deconv_256_Conv',
                                             kernel_regularizer=l2(l2_weight))(x_deconv128)
                return x_deconv256

            # in detnet setting, C6.shape == C5.shape == C4.shape, in fcn27 this is 1/4 of origin image
            C7_deconv = _32to256(C7, 'C7')
            C6_deconv = _32to256(C6, 'C6')
            C5_deconv = _32to256(C5, 'C5')
            C4_deconv_128 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=128, strides=(2, 2), name='C4_deconv_128_Conv',
                                             kernel_regularizer=l2(l2_weight))(C4)
            C4_deconv_128 = BatchNormalization(name='C4_deconv_128_BN')(C4_deconv_128)
            C4_deconv_128 = Activation('relu', name='C4_deconv_128_RELU')(C4_deconv_128)
            C4_deconv_256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=128, strides=(2, 2), name='C4_deconv_256_Conv',
                                             kernel_regularizer=l2(l2_weight))(C4_deconv_128)


            C3_deconv_256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=64, strides=(2, 2), name='C3_deconv_256_Conv',
                                             kernel_regularizer=l2(l2_weight))(C3)

            C23456_concat = Concatenate()([C7_deconv, C6_deconv, C5_deconv, C4_deconv_256, C3_deconv_256])

            return C23456_concat

        # tf.reset_default_graph()
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input, l2_weight=l2_weight)
        x_stage2, x_stage3, x_stage4, x_stage5 = fcn_36(x_stage1, l2_weight=l2_weight)
        # stage3 is 1/2 size
        #########
        # following layer proposed by DetNet
        #########
        x_stage6_B = dilated_with_projection(x_stage5, stage=6)
        x_stage6_A1 = dilated_bottleneck(x_stage6_B, stage=6, block=1)
        x_stage6 = dilated_bottleneck(x_stage6_A1, stage=6, block=2)
        x_stage7_B = dilated_with_projection(x_stage6, stage=7)
        x_stage7_A1 = dilated_bottleneck(x_stage7_B, stage=7, block=1)
        x_stage7 = dilated_bottleneck(x_stage7_A1, stage=7, block=2)

        ########
        # 1x1 convolutnion part
        ########
        #x_stage2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='stage2_1x1_conv')(x_stage2)
        x_stage3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                              name='stage3_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage3)
        x_stage4_1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                              name='stage4_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage4)
        # stage5 is 1/8 size, same as 6 and 7
        x_stage5_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv')(x_stage5)
        x_stage6_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv')(x_stage6)
        x_stage7_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage7_1x1_conv')(x_stage7)

        stage_67 = Add(name='add_stage_6_7')([x_stage6_1x1, x_stage7_1x1])
        stage_67 = stage_67
        stage_567 = Add(name='add_stage_5_6_7')([x_stage5_1x1, stage_67])
        stage_567_upsample = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2),padding='same',
                                             name='stage_567_upsample',
                                             kernel_regularizer=l2(l2_weight))(stage_567)
        stage_4567 = Add(name='add_stage_4_567')([stage_567_upsample, x_stage4_1x1])
        stage_4567_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              name='stage_4567_upsample')(stage_4567)
        stage_34567 = Add(name='add_stage_3_4567')([stage_4567_upsample, x_stage3_1x1])  # filters = 64
        #stage_34567_upsample = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', name='stage_34567_upsample')(stage_34567)
        #stage_234567 = Add(name='add_stage_2_34567')([stage_34567_upsample, x_stage2_1x1])

        x_feature_concat= _feature_concat_deconv_branch(C7=x_stage7_1x1, C6=stage_67, C5=stage_567,
                                                        C4=stage_4567, C3=stage_34567)
        x_feature_concat = Conv2D(kernel_size=(1,1), filters=2,padding='same',
                                  name='all_feature_concat')(x_feature_concat)
        x_output = Activation('softmax', name='Final_Softmax')(x_feature_concat)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model

    def relufirst_fcn36_deconv_backbone(self, l2_weight=0.001):
        #######################################
        # Extra branch for every pyramid feature
        #######################################
        def _feature_concat_deconv_branch(C7=None, C6=None, C5=None, C4=None, C3=None):
            """

            :param features: input feature is from every feature pyramid layer,
                             should've been already connect with 1x1 convolution layer.
            """

            def _32to256(input, type):
                x_deconv64 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_64_Conv',
                                             padding='same')(input)
                x_deconv64 = BatchNormalization(name=type + '_deconv_64_BN')(x_deconv64)
                x_deconv64 = Activation('relu', name=type + '_deconv_64_RELU')(x_deconv64)
                x_deconv128 = Conv2DTranspose(kernel_size=(3, 3),
                                              filters=256, strides=(2, 2), name=type + '_deconv_128_Conv',
                                              padding='same')(x_deconv64)
                x_deconv128 = BatchNormalization(name=type + '_deconv_128_BN')(x_deconv128)
                x_deconv128 = Activation('relu', name=type + '_deconv_128_RELU')(x_deconv128)
                x_deconv256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                              filters=256, strides=(2, 2), name=type + '_deconv_256_Conv')(x_deconv128)
                x_deconv256 = BatchNormalization()(x_deconv256)
                return x_deconv256

            # in detnet setting, C6.shape == C5.shape == C4.shape, in fcn27 this is 1/4 of origin image
            C7_deconv = _32to256(C7, 'C7')
            C6_deconv = _32to256(C6, 'C6')
            C5_deconv = _32to256(C5, 'C5')
            C4_deconv_128 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=128, strides=(2, 2), name='C4_deconv_128_Conv')(C4)
            C4_deconv_128 = BatchNormalization(name='C4_deconv_128_BN')(C4_deconv_128)
            C4_deconv_128 = Activation('relu', name='C4_deconv_128_RELU')(C4_deconv_128)
            C4_deconv_256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=128, strides=(2, 2), name='C4_deconv_256_Conv')(C4_deconv_128)
            C4_deconv_256 = BatchNormalization()(C4_deconv_256)

            C3_deconv_256 = Conv2DTranspose(kernel_size=(3, 3), padding='same',
                                            filters=64, strides=(2, 2), name='C3_deconv_256_Conv')(C3)
            C3_deconv_256 = BatchNormalization()(C3_deconv_256)
            C23456_concat = Concatenate()([C7_deconv, C6_deconv, C5_deconv, C4_deconv_256, C3_deconv_256])

            return C23456_concat

        # tf.reset_default_graph()
        img_input = Input(self.input_shape)
        #########
        # Adapted first stage
        #########
        x_stage1 = self.first_layer(inputs=img_input, l2_weight=l2_weight)
        x_stage2, x_stage3, x_stage4, x_stage5 = fcn_36(x_stage1, l2_weight=l2_weight)
        # stage3 is 1/2 size
        #########
        # following layer proposed by DetNet
        #########
        x_stage6_B = dilated_with_projection(x_stage5, stage=6)
        x_stage6_A1 = dilated_bottleneck(x_stage6_B, stage=6, block=1)
        x_stage6 = dilated_bottleneck(x_stage6_A1, stage=6, block=2)
        x_stage7_B = dilated_with_projection(x_stage6, stage=7)
        x_stage7_A1 = dilated_bottleneck(x_stage7_B, stage=7, block=1)
        x_stage7 = dilated_bottleneck(x_stage7_A1, stage=7, block=2)

        ########
        # 1x1 convolutnion part
        ########
        #x_stage2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='stage2_1x1_conv')(x_stage2)
        x_stage3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                              name='stage3_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage3)
        x_stage4_1x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                              name='stage4_1x1_conv',
                              kernel_regularizer=l2(l2_weight))(x_stage4)
        # stage5 is 1/8 size, same as 6 and 7
        x_stage5_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage5_1x1_conv')(x_stage5)
        x_stage6_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage6_1x1_conv')(x_stage6)
        x_stage7_1x1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same',
                              name='stage7_1x1_conv')(x_stage7)

        x_stage7_1x1 = BatchNormalization(name='stage7_1x1_BN')(x_stage7_1x1)
        x_stage6_1x1 = BatchNormalization(name='stage6_1x1_BN')(x_stage6_1x1)
        x_stage5_1x1 = BatchNormalization(name='stage5_1x1_BN')(x_stage5_1x1)
        x_stage4_1x1 = BatchNormalization(name='stage4_1x1_BN')(x_stage4_1x1)
        x_stage3_1x1 = BatchNormalization(name='stage3_1x1_BN')(x_stage3_1x1)


        stage_67 = Add(name='add_stage_6_7')([x_stage6_1x1, x_stage7_1x1])
        #stage_67 = Activation('relu', name='stage67_relu')(stage_67)
        stage_567 = Add(name='add_stage_5_6_7')([x_stage5_1x1, stage_67])
        stage_567 = Activation('relu', name='stage67_relu')(stage_67)
        stage_567_upsample = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2),padding='same',
                                             name='stage_567_upsample',
                                             kernel_regularizer=l2(l2_weight))(stage_567)
        stage_567_upsample = BatchNormalization(name='stage_567_upsample_BN')(stage_567_upsample)
        #stage_567_upsample = Activation('relu')(stage_567_upsample)


        stage_4567 = Add(name='add_stage_4_567')([stage_567_upsample, x_stage4_1x1])
        stage_4567 = Activation('relu')(stage_4567)
        stage_4567_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                              padding='same',
                                              kernel_regularizer=keras.regularizers.l2(l2_weight),
                                              name='stage_4567_upsample')(stage_4567)
        stage_4567_upsample = BatchNormalization()(stage_4567_upsample)
        stage_34567 = Add(name='add_stage_3_4567')([stage_4567_upsample, x_stage3_1x1])  # filters = 64
        stage_34567 = Activation('relu')(stage_34567)

        x_feature_concat = _feature_concat_deconv_branch(C7=x_stage7_1x1, C6=stage_67, C5=stage_567,
                                                        C4=stage_4567, C3=stage_34567)
        x_feature_concat = Conv2D(kernel_size=(1,1), filters=5,padding='same',
                                  name='all_feature_concat')(x_feature_concat)
        x_feature_concat = BatchNormalization()(x_feature_concat)
        x_output = Activation('softmax', name='Final_Softmax')(x_feature_concat)
        detnet_model = Model(inputs=img_input,
                             outputs=x_output)
        return detnet_model




def fcn_tune_loss_weight():
    """
    use this function to fine tune weights later.
    :return:
    """
    print('weight initialized')
    det_weight = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    l2_weight = [0.0011, 0.0015, 0.002, 0.0025,0.003, 0.0035, 0.004, 0.0045, 0.0050, 0.0055, 0.006, 0.0065, 0.007]

    fkg_smooth_factor = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    bkg_smooth_factor = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    ind_factor = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    smooth_factor1 = [0.8, 0.9, 1.0]
    smooth_factor2 = [0.8, 0.9, 1.0, 1.1]
    smooth_factor3 = [0.5, 0.6, 0.7, 0.8]
    smooth_factor4 = [1.7, 1.8, 1.9, 2.0]
    return [det_weight, fkg_smooth_factor, l2_weight, bkg_smooth_factor, ind_factor,
            smooth_factor1, smooth_factor2, smooth_factor3, smooth_factor4]


def data_prepare(print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """
    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    print("croppppppppppppppppppp")
    #train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train')
    #valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation')
    #test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test')
    train_imgs, train_det_masks, train_cls_masks = load_dataset(dataset=DATA_DIR, type='train', reshape_size=(256, 256))
    valid_imgs, valid_det_masks, valid_cls_masks = load_dataset(dataset=DATA_DIR, type='validation', reshape_size=(256, 256))
    test_imgs, test_det_masks, test_cls_masks = load_dataset(dataset=DATA_DIR, type='test',reshape_size=(256, 256))

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_det_masks: {}, train_cls_masks: {}'.format(train_imgs.shape, train_det_masks.shape, train_cls_masks.shape))
        print('valid_imgs: {}, valid_det_masks: {}, valid_cls_masks: {}'.format(valid_imgs.shape, valid_det_masks.shape, valid_cls_masks.shape))
        print('test_imgs: {}, test_det_masks: {}'.format(test_imgs.shape, test_det_masks.shape))
        print()

    train_det = np_utils.to_categorical(train_det_masks, 2)
    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    print('train_det: {}'.format(train_det.shape))
    #train_det = reshape_mask(train_det_masks, train_det, 2)

    valid_det = np_utils.to_categorical(valid_det_masks, 2)
    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)
    #valid_det = reshape_mask(valid_det_masks, valid_det, 2)

    test_det = np_utils.to_categorical(test_det_masks, 2)
    #test_det = reshape_mask(test_det_masks, test_det, 2)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}, train_cls: {}'.format(train_imgs.shape, train_det.shape, train_cls.shape))
        print('valid_imgs: {}, valid_det: {}, valid_cls: {}'.format(valid_imgs.shape, valid_det.shape, valid_cls.shape))
        print('test_imgs: {}, test_det: {}'.format(test_imgs.shape, test_det.shape))
        print()
    return [train_imgs, train_det, valid_imgs, valid_det, train_cls, valid_cls]


def fcn_detnet_focal_model_compile(nn, det_loss_weight,
                         optimizer, summary=False,
                         fkg_smooth_factor=None,
                         bkg_smooth_factor=None):

    loss_input = detection_double_focal_loss_K(det_loss_weight,
                                               fkg_smooth_factor,
                                               bkg_smooth_factor)
    nn.compile(optimizer=optimizer,
                      loss=loss_input,
                      metrics=['accuracy'])
    if summary==True:
        nn.summary()
    return nn


def cls_focal_compile(nn, cls_loss_weight,
                         optimizer, summary=False,
                         fkg_smooth_factor=None,
                         bkg_smooth_factor=None):

    loss_input = cls_cross_entropy(cls_loss_weight)
    nn.compile(optimizer=optimizer,
                      loss=loss_input,
                      metrics=['accuracy'])
    if summary==True:
        nn.summary()
    return nn


def fcn_detnet_normal_model_compile(nn, det_loss_weight,
                         optimizer, summary=False):

    loss_input = detection_loss_K(det_loss_weight)
    nn.compile(optimizer=optimizer,
                      loss=loss_input,
                      metrics=['accuracy'])
    if summary==True:
        nn.summary()
    return nn


def multi_callback_preparation(model, hyper):
    """
    implement necessary callbacks into model.
    :return: list of callback.
    """
    #timer.set_model(model)
    tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper +'_tb_logs'))
    checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR,
                                                       hyper + '_cp.h5'), period=1)
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       min_delta=0.001)
    return [tensorboard_callback, earlystop_callback]


def set_fcn36_num_step_and_aug():
    """
    Because the size of image is big and it would store in computation graph for doing back propagation,
    we set different augmentation number and training step depends on which struture we are using.
    :return:
    """
    NUM_TO_AUG,TRAIN_STEP_PER_EPOCH = 3, 135
    return NUM_TO_AUG, TRAIN_STEP_PER_EPOCH

def generator(features, det_labels, batch_size):
    batch_features = np.zeros((batch_size, 256, 256, 3))
    batch_det_labels = np.zeros((batch_size, 256, 256, 5))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            feature_index = features[index]
            det_label_index = det_labels[index]
            batch_features[counter] = feature_index
            batch_det_labels[counter] = det_label_index
            counter = counter + 1
        yield batch_features, batch_det_labels

class Score(Callback):
    """Tracking time spend on each epoch as well as whole training process.
    """
    def __init__(self):
        super(Score, self).__init__()
        self.fscore = 0
        self.precision = 0


    def on_epoch_end(self, epoch, logs=None):
        with tf.device('/cpu:0'):
            print('test results: ')
            p, r, f = mymetrics(self.model)
            print('P: {}, R: {}, F1: {}'.format(p, r, f))


if __name__ == '__main__':
   # if Config.gpu_count == 1:
       # os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu1
    fcn_detnet = Fcn_det().relufirst_fcn36_deconv_backbone()
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   min_delta=0.001)
    hyper_para = fcn_tune_loss_weight()
    BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
    EPOCHS = Config.epoch
    data = data_prepare(print_input_shape=True, print_image_shape=True)
    optimizer = SGD(lr=0.01, decay=0.00001, momentum=0.9, nesterov=True)
    STEP_PER_EPOCH = int(len(data[0]) / BATCH_SIZE)
    print('batch size is :', BATCH_SIZE)
    if Config.gpu_count != 1:
        multi_gpu_fcn_detnet = keras.utils.multi_gpu_model(fcn_detnet, gpus=Config.gpu_count)
        for i, epi_weight in enumerate(hyper_para[5]):
            for j, fib_weight in enumerate(hyper_para[6]):
                for x, inf_weight in enumerate(hyper_para[7]):
                    for v, other_weight in enumerate(hyper_para[8]):
                        hyper = '{}_epi:{}_fib:{}_inf:{}_other:{}_lr:0.01'.format('cls_multi_celld',
                                                                                     epi_weight, fib_weight,
                                                                                     inf_weight, other_weight)

                        tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper + '_tb_logs'))
                        timer = TimerCallback()
                        print(hyper)
                        print()
                        model_weights_saver = os.path.join(WEIGHTS_DIR, hyper + '_train.h5')
                        cls_loss = np.array([0.5, i, j, x, v])
                        if not os.path.exists(model_weights_saver):
                            fcn_detnet_model = cls_focal_compile(nn=multi_gpu_fcn_detnet,cls_loss_weight=cls_loss, optimizer=optimizer)
                            print('{} gpu fcn36 focal detection is training'.format(Config.gpu_count))

                            #list_callback = callback_preparation(fcn_detnet, hyper)
                            #list_callback.append(LearningRateScheduler(lr_scheduler))
                            score = Score()
                            #score.set_model(fcn_detnet)
                            #list_callback.append(Score)
                            fcn_detnet_model.fit_generator(generator(data[0],
                                                                     data[4],
                                                                     batch_size=BATCH_SIZE),
                                                           epochs=EPOCHS,
                                                           steps_per_epoch=STEP_PER_EPOCH,
                                                           validation_data=generator(
                                                               data[2], data[5], batch_size=BATCH_SIZE),
                                                           validation_steps=3,
                                                           callbacks=[timer, tensorboard_callback, earlystop_callback,
                                                                      LearningRateScheduler(lr_scheduler)])
                            model_json = fcn_detnet.to_json()
                            with open('json_' + hyper + '.json', 'w') as json_file:
                                json_file.write(model_json)
                            fcn_detnet.save_weights(model_weights_saver)
                            print(hyper + 'has been saved')
    else:
        print('------------------------------------')
        print('This model is using {}'.format(Config.backbone))
        print()
        hyper_para = fcn_tune_loss_weight()
        BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
        print('batch size is :', BATCH_SIZE)
        EPOCHS = Config.epoch

        data = data_prepare(print_input_shape=True, print_image_shape=True)
        optimizer = SGD(lr=0.01, decay=0.00001, momentum=0.9, nesterov=True)
        STEP_PER_EPOCH = int(len(data[0]) / BATCH_SIZE)

        for j, fkg_weight in enumerate(hyper_para[1]):
            for k, bkg_weight in enumerate(hyper_para[3]):
                hyper = '{}_loss:{}_det:{}_fkg:{}_bkg:{}_lr:0.01'.format('multi_fcn36', 'fd',
                                                                         Config.det_weight,
                                                                         fkg_weight,
                                                                         bkg_weight)  # _l2:{}_bkg:{}'.format()
                tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper + '_tb_logs'))
                timer = TimerCallback()
                print(hyper)
                print()
                model_weights_saver = os.path.join(WEIGHTS_DIR, hyper + '_train.h5')

                if not os.path.exists(model_weights_saver):
                    det_weight2 = 1.0 - Config.det_weight
                    print()
                    print('........the det_weight2 is : ', det_weight2)
                    print()
                    fcn_detnet_model = fcn_detnet_focal_model_compile(nn=fcn_detnet,
                                                                      summary=Config.summary,
                                                                      det_loss_weight=np.array(
                                                                          [Config.det_weight, det_weight2]),
                                                                      optimizer=optimizer,
                                                                      fkg_smooth_factor=fkg_weight,
                                                                      bkg_smooth_factor=bkg_weight)
                    print('{} gpu fcn36 focal detection is training'.format(Config.gpu_count))

                    # list_callback = callback_preparation(fcn_detnet, hyper)
                    # list_callback.append(LearningRateScheduler(lr_scheduler))
                    score = Score()
                    # score.set_model(fcn_detnet)
                    # list_callback.append(Score)

                    fcn_detnet_model.fit_generator(generator(data[0],
                                                             data[1],
                                                             batch_size=BATCH_SIZE),
                                                   epochs=EPOCHS,
                                                   steps_per_epoch=STEP_PER_EPOCH,
                                                   validation_data=generator(
                                                       data[2], data[3], batch_size=BATCH_SIZE),
                                                   validation_steps=3,
                                                   callbacks=[timer, tensorboard_callback, earlystop_callback,
                                                              LearningRateScheduler(lr_scheduler)])
                    model_json = fcn_detnet.to_json()
                    with open('json_' + hyper + '.json', 'w') as json_file:
                        json_file.write(model_json)
                    fcn_detnet.save_weights(model_weights_saver)
                    print(hyper + 'has been saved')