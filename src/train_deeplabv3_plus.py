from util import load_data, set_gpu, set_num_step_and_aug, lr_scheduler, aug_on_fly, heavy_aug_on_fly
from keras.optimizers import SGD
from encoder_decoder_object_det import data_prepare, tune_loss_weight, TimerCallback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from config import Config
from keras.utils import np_utils
from deeplabv3_plus import load_pretrain_weights, preprocess_input, Deeplab, Deeplabv3Plus
from loss import deeplab_cls_loss
import os
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import keras
import scipy.misc as misc

weight_decay = 0.005
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')
CROP_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
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


def callback_preparation(model, hyper):
    """
    implement necessary callbacks into model.
    :return: list of callback.
    """
    timer = TimerCallback()
    timer.set_model(model)
    tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper +'_tb_logs'))
    #checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR,
                                                     #  hyper + '_cp.h5'), period=1)
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       min_delta=0.001)
    return [tensorboard_callback, timer, earlystop_callback]


def generator(features, det_labels, batch_size):
    batch_features = np.zeros((batch_size, 256, 256, 3))
    batch_det_labels = np.zeros((batch_size, 256, 256, 2))
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


def data_prepare(print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """
    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train', det=False, cls=True)
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation', det=False, cls=True)
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test',det=False, cls=True)

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_cls_masks: {}'.format(train_imgs.shape, train_cls_masks.shape))
        print('valid_imgs: {}, valid_cls_masks: {}'.format(valid_imgs.shape, valid_cls_masks.shape))
        print('test_imgs: {}, test_cls_masks: {}'.format(test_imgs.shape, test_cls_masks.shape))
        print()

    train_cls = np_utils.to_categorical(train_cls_masks, 5)
    train_cls = reshape_mask(train_cls_masks, train_cls, 5)

    valid_cls = np_utils.to_categorical(valid_cls_masks, 5)
    valid_cls = reshape_mask(valid_cls_masks, valid_cls, 5)

    test_cls = np_utils.to_categorical(test_cls_masks, 5)
    test_cls = reshape_mask(test_cls_masks, test_cls, 5)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}'.format(train_imgs.shape, train_cls.shape))
        print('valid_imgs: {}, valid_det: {}'.format(valid_imgs.shape, valid_cls.shape))
        print('test_imgs: {}, test_det: {}'.format(test_imgs.shape, test_cls.shape))
        print()
    return [train_imgs, train_cls, valid_imgs, valid_cls,  test_imgs, test_cls]


def fcn_tune_loss_weight():
    """
    use this function to fine tune weights later.
    :return:
    """
    print('weight initialized')
    det_weight = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    l2_weight = 0.001

    fkg_smooth_factor = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    bkg_smooth_factor = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    ind_factor = [np.array([0.2, 0.8]),np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    return [det_weight, fkg_smooth_factor, l2_weight, bkg_smooth_factor, ind_factor]


if __name__ == '__main__':
   # if Config.gpu_count == 1:
       # os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu1
    print('------------------------------------')
    print('This model is using {}'.format(Config.backbone))
    print()
    hyper_para = fcn_tune_loss_weight()
    BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
    print('batch size is :', BATCH_SIZE)
    EPOCHS = Config.epoch
    network = Deeplabv3Plus(backbone=Config.backbone, input_shape=(256, 256, 3), classes=5)
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   min_delta=0.001)
    if Config.gpu_count != 1:
        network = keras.utils.multi_gpu_model(network, gpus=Config.gpu_count)

    data = data_prepare(print_input_shape=True, print_image_shape=True)
    optimizer = SGD(lr=0.01, decay=0.00001, momentum=0.9, nesterov=True)
    STEP_PER_EPOCH = int(len(data[0])/BATCH_SIZE)

    #for k, bkg_weight in enumerate(hyper_para[3]):
        #for j, fkg_weight in enumerate(hyper_para[1]):
    hyper = '{}_{}_loss:{}_lr:0.01'.format('Deeplabv3', Config.backbone, 'categorical_crossentropy')  # _l2:{}_bkg:{}'.format()
    tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, hyper + '_tb_logs'))
    timer = TimerCallback()
    print(hyper)
    print()
    model_weights_saver = os.path.join(WEIGHTS_DIR, hyper + '_train.h5')
    network.summary()
    if not os.path.exists(model_weights_saver):
        print('model start to compile')
        network.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])

        print('{} gpu classification is training'.format(Config.gpu_count))


        network.fit_generator(generator(data[0], data[1], batch_size=BATCH_SIZE),
                                        epochs=EPOCHS,
                                        steps_per_epoch=STEP_PER_EPOCH,
                                        validation_data=generator(data[2], data[3], batch_size=BATCH_SIZE),
                                        validation_steps=3,
                                        callbacks=[timer, tensorboard_callback, earlystop_callback,
                                                   LearningRateScheduler(lr_scheduler)])
        model_json = network.to_json()
        with open('json_' + hyper + '.json', 'w') as json_file:
            json_file.write(model_json)
        network.save_weights(model_weights_saver)
        print(hyper + 'has been saved')
