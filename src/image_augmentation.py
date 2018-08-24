import os
import cv2
from numpy.random import randint
import numpy as np
from imgaug import augmenters as iaa
from util import check_directory,check_cv2_imwrite
from imgaug import augmenters as iaa
import imgaug as ia

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

OLD_DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
TRAIN_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'train')
TEST_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'test')
VALID_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'validation')
CROP_TARGET_DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'crop_cls_and_det')
CROP_TRAIN_TARGET_DATA_DIR = os.path.join(CROP_TARGET_DATA_DIR, 'train')
CROP_TEST_TARGET_DATA_DIR = os.path.join(CROP_TARGET_DATA_DIR, 'test')
CROP_VALID_TARGET_DATA_DIR = os.path.join(CROP_TARGET_DATA_DIR, 'validation')

TARGET_DATA_DIR = os.path.join(ROOT_DIR, 'aug')
TRAIN_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'train')
TEST_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'test')
VALID_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'validation')


def crop_image_parts(image, det_mask, cls_mask, origin_shape=(512, 512)):
    assert image.ndim == 3
    ori_width, ori_height = origin_shape[0], origin_shape[1]
    des_width, des_height = 256, 256
    assert des_width == des_height

    cropped_img1 = image[0: des_width, 0: des_height, :]  # 1, 3
    cropped_img2 = image[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_img3 = image[0: des_width, des_height: ori_height, :]
    cropped_img4 = image[des_width: ori_width, des_height: ori_height, :]

    cropped_det_mask1 = det_mask[0: des_width, 0: des_height, :]              # 1, 3
    cropped_det_mask2 = det_mask[des_width: ori_width, 0: des_height, :]      # 2, 4
    cropped_det_mask3 = det_mask[0: des_width, des_height: ori_height, :]
    cropped_det_mask4 = det_mask[des_width: ori_width, des_height: ori_height, :]

    cropped_cls_mask1 = cls_mask[0: des_width, 0: des_height, :]  # 1, 3
    cropped_cls_mask2 = cls_mask[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_cls_mask3 = cls_mask[0: des_width, des_height: ori_height, :]
    cropped_cls_mask4 = cls_mask[des_width: ori_width, des_height: ori_height, :]

    return [cropped_img1, cropped_img2, cropped_img3, cropped_img4,
            cropped_det_mask1, cropped_det_mask2, cropped_det_mask3, cropped_det_mask4,
            cropped_cls_mask1, cropped_cls_mask2, cropped_cls_mask3, cropped_cls_mask4,
            cropped_img1, cropped_img2, cropped_img3, cropped_img4]


def batch_image_process(ori_set, target_set, process, stage=None, num_op=None):
    """

    :param ori_set:
    :param target_set:
    :param process: process can be 'crop' or 'flip'
    :return:
    """
    check_directory([TARGET_DATA_DIR, TEST_TARGET_DATA_DIR, TRAIN_TARGET_DATA_DIR, VALID_TARGET_DATA_DIR])
    check_directory([CROP_TARGET_DATA_DIR, CROP_TEST_TARGET_DATA_DIR, CROP_TRAIN_TARGET_DATA_DIR, CROP_VALID_TARGET_DATA_DIR])
    for file in os.listdir(ori_set):
        print(file)
        image_file = os.path.join(ori_set, str(file), str(file) + '.bmp')
        det_mask_file = os.path.join(ori_set, str(file), str(file) + '_detection.bmp')
        cls_mask_file = os.path.join(ori_set, str(file), str(file) + '_classification.bmp')

        image = cv2.imread(image_file)
        det_mask = cv2.imread(det_mask_file)
        cls_mask = cv2.imread(cls_mask_file)
        if process == 'crop':
            image = cv2.resize(image, (512, 512))
            det_mask = cv2.resize(det_mask, (512, 512))
            cls_mask = cv2.resize(cls_mask, (512, 512))
            crop_list = crop_image_parts(image, det_mask, cls_mask)

            list_file_create = [os.path.join(target_set, str(file)+'_1'),
                                os.path.join(target_set, str(file)+'_2'),
                                os.path.join(target_set, str(file)+'_3'),
                                os.path.join(target_set, str(file)+'_4')]
            check_directory(list_file_create)
            list_img_create = [os.path.join(target_set, str(file)+ '_1', str(file)+'_1.bmp'),
                               os.path.join(target_set, str(file)+ '_2', str(file)+'_2.bmp'),
                               os.path.join(target_set, str(file)+ '_3', str(file)+'_3.bmp'),
                               os.path.join(target_set, str(file)+'_4', str(file)+'_4.bmp'),
                               os.path.join(target_set, str(file)+'_1',str(file)+'_1_detection.bmp'),
                               os.path.join(target_set, str(file)+'_2',str(file)+'_2_detection.bmp'),
                               os.path.join(target_set, str(file)+'_3',str(file)+'_3_detection.bmp'),
                               os.path.join(target_set, str(file)+'_4',str(file)+'_4_detection.bmp'),
                               os.path.join(target_set, str(file) + '_1', str(file) + '_1_classification.bmp'),
                               os.path.join(target_set, str(file) + '_2', str(file) + '_2_classification.bmp'),
                               os.path.join(target_set, str(file) + '_3', str(file) + '_3_classification.bmp'),
                               os.path.join(target_set, str(file) + '_4', str(file) + '_4_classification.bmp'),
                               os.path.join(target_set, str(file) + '_1', str(file) + '_1_original.bmp'),
                               os.path.join(target_set, str(file) + '_2', str(file) + '_2_original.bmp'),
                               os.path.join(target_set, str(file) + '_3', str(file) + '_3_original.bmp'),
                               os.path.join(target_set, str(file) + '_4', str(file) + '_4_original.bmp')]
            for order, img in enumerate(crop_list):
                check_cv2_imwrite(list_img_create[order], img)

        elif process == 'flip':
            aug_lrimage, aug_lrdetmask, aug_lrclsmask = img_aug(image, det_mask, cls_mask, 'fliplr')
            aug_upimage, aug_updetmask, aug_upclsmask = img_aug(image, det_mask, cls_mask, 'flipup')
            list_file_create = [os.path.join(target_set, str(file) + '_lr'),
                                os.path.join(target_set, str(file) + '_up')]
            check_directory(list_file_create)
            list_img_create = [os.path.join(target_set, str(file) + '_lr', str(file) + '_lr.bmp'),
                               os.path.join(target_set, str(file) + '_up', str(file) + '_up.bmp'),
                               os.path.join(target_set, str(file) + '_lr', str(file) + '_lr_detection.bmp'),
                               os.path.join(target_set, str(file) + '_up', str(file) + '_up_detection.bmp'),
                               os.path.join(target_set, str(file) + '_lr', str(file) + '_lr_classification.bmp'),
                               os.path.join(target_set, str(file) + '_up', str(file) + '_up_classification.bmp'),
                               os.path.join(target_set, str(file) + '_lr', str(file) + '_lr_original.bmp'),
                               os.path.join(target_set, str(file) + '_up', str(file) + '_up_original.bmp')]
            flip_list = [aug_lrimage, aug_upimage,
                        aug_lrdetmask, aug_updetmask,
                        aug_lrclsmask, aug_upclsmask,
                        aug_lrimage, aug_upimage]
            for order, img in enumerate(flip_list):
                check_cv2_imwrite(list_img_create[order], img)

        elif process == 'shear' or process == 'channel' or process == 'random' or process=='zoom' or process=='rotate':
            aug_image, aug_detmask, aug_clsmask = img_aug(image, det_mask, cls_mask, process)
            list_file_create = [os.path.join(target_set, str(file) + '_' + process + str(stage))]
            check_directory(list_file_create)
            list_img_create = [os.path.join(target_set, str(file) + '_' + process + str(stage), str(file) + '_' + process + str(stage) + '.bmp'),
                               os.path.join(target_set, str(file) + '_' + process + str(stage), str(file) + '_' + process + str(stage) + '_detection.bmp'),
                               os.path.join(target_set, str(file) + '_' + process + str(stage), str(file) + '_' + process + str(stage) + '_classification.bmp'),
                               os.path.join(target_set, str(file) + '_' + process + str(stage), str(file) + '_' + process + str(stage) + '_original.bmp')]
            image_list = [aug_image, aug_detmask, aug_clsmask, aug_image]
            for order, img in enumerate(image_list):
                check_cv2_imwrite(list_img_create[order], img)
        else:
            raise ValueError('augmentation type wrong, check documents.')


def initialize_train_test_folder(file_path):
    """
    create new train, test and validation folder if
    it doesn't exist.
    :param file_path: main_path
    """
    train_folder = '{}/{}'.format(file_path, 'train')
    test_folder = '{}/{}'.format(file_path, 'test')
    valid_folder = '{}/{}'.format(file_path, 'validation')
    check_directory(train_folder)
    check_directory(test_folder)
    check_directory(valid_folder)


def img_aug(img, det_mask, cls_mask, aug_type):
    """Do augmentation with different combination on each training batch
    """
    def _det_aug(image, det_mask, cls_mask, seq):
        seq = seq.to_deterministic()
        aug_img = seq.augment_image(image)
        aug_det_mask = seq.augment_image(det_mask)
        aug_cls_mask = seq.augment_image(cls_mask)
        return [aug_img, aug_det_mask, aug_cls_mask]

    def iaa_augmentation(image, masks, aug_type):
        # without additional operations
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        fliplr = iaa.Sequential([
            iaa.Fliplr(1)
        ])
        flipup = iaa.Sequential([
            iaa.Flipud(1)
        ])
        rotate = iaa.Sequential([
            iaa.Affine(rotate=(-90, 90))
        ])
        shear = iaa.Sequential([
            iaa.Affine(shear=(-16, 16))
        ])
        zoom = iaa.Sequential([
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        ])
        channel_shift = iaa.Sequential([
            iaa.PerspectiveTransform(scale=(0.01, 0.1))
        ])
        random = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Fliplr(1),
                iaa.Flipud(1),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)}),
                #iaa.PerspectiveTransform(scale=(0.01, 0.1))
            ])
        ])
        det_mask, cls_mask = masks[0], masks[1]
        if aug_type == 'fliplr':
            aug = _det_aug(image, det_mask, cls_mask, fliplr)
        elif aug_type == 'flipup':
            aug = _det_aug(image, det_mask, cls_mask, flipup)
        elif aug_type == 'rotate':
            aug = _det_aug(image, det_mask, cls_mask, rotate)
        elif aug_type == 'shear':
            aug = _det_aug(image, det_mask, cls_mask, shear)
        elif aug_type == 'zoom':
            aug = _det_aug(image, det_mask, cls_mask, zoom)
        elif aug_type == 'channel':
            aug = _det_aug(image, det_mask, cls_mask, channel_shift)
        elif aug_type == 'random':
            aug = _det_aug(image, det_mask, cls_mask, random)

        return aug[0], aug[1], aug[2]

    aug_image, aug_det_mask, aug_cls_mask = iaa_augmentation(image=img,
                                                             masks=[det_mask, cls_mask],
                                                             aug_type=aug_type)
    return aug_image, aug_det_mask, aug_cls_mask


class ImageCropping:
    def __init__(self, data_path = None, old_filename = None, new_filename = None):
        self.data_path = data_path
        self.old_filename = '{}/{}'.format(data_path, old_filename)
        self.new_filename = '{}/{}'.format(data_path, new_filename)
        check_directory(self.new_filename)
        initialize_train_test_folder(self.new_filename)

    @staticmethod
    def crop_image_batch(image, masks=None, if_mask=True, if_det=True, if_cls=True,
                         origin_shape=(500, 500), desired_shape=(64, 64)):
        assert image.ndim == 4
        ori_width, ori_height = origin_shape[0], origin_shape[1]
        des_width, des_height = desired_shape[0], desired_shape[1]

        max_x = ori_width - des_width
        max_y = ori_height - des_height
        ran_x = np.random.randint(0, max_x)
        ran_y = np.random.randint(0, max_y)
        cropped_x = ran_x + des_width
        cropped_y = ran_y + des_height
        cropped_img = image[:, ran_x:cropped_x, ran_y:cropped_y]
        if if_mask and masks is not None:
            if if_det and if_cls:
                det_mask = masks[0]
                cls_mask = masks[1]
                cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask, cropped_cls_mask
            elif if_det and not if_cls:
                det_mask = masks
                cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask
            elif if_cls and not if_det:
                cls_mask = masks
                cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y, :]
                return cropped_img, cropped_cls_mask
        else:
            return cropped_img

    @staticmethod
    def ran_crop_image(image, masks=None, if_mask=True, if_det = True, if_cls = True,
                   origin_shape=(500, 500), desired_shape=(64, 64)):
        assert image.ndim == 3
        ori_width, ori_height = origin_shape[0], origin_shape[1]
        des_width, des_height = desired_shape[0], desired_shape[1]

        max_x = ori_width - des_width
        max_y = ori_height - des_height
        ran_x = randint(0, max_x)
        ran_y = randint(0, max_y)
        cropped_x = ran_x + des_width
        cropped_y = ran_y + des_height
        cropped_img = image[ran_x:cropped_x, ran_y:cropped_y]
        if if_mask and masks is not None:
            if if_det and if_cls:
                det_mask = masks[0]
                cls_mask = masks[1]
                cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
                cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask, cropped_cls_mask
            elif if_det and not if_cls:
                det_mask = masks[0]
                cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask
            elif if_cls and not if_det:
                cls_mask = masks[0]
                cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_cls_mask
        else:
            return cropped_img


def copy_oriimg_to_new(oriset, targetset):
    """
    After augmentation, copy image files from original files into new files.
    :param oriset:
    :param targetset:
    :return:
    """
    from shutil import copytree, copyfile
    for file in os.listdir(oriset):
        old_path = os.path.join(oriset, file)
        # rename image.bmp to image_original.bmp for
        # the convenience of future recognation.
        for img_file in os.listdir(old_path):
            if img_file == str(file) + '.bmp':
                new_img_path = os.path.join(old_path, str(file) + '_original.bmp')
                copyfile(os.path.join(old_path, img_file), new_img_path)
        # move old file into new dataset files.
        new_path = os.path.join(targetset, file)
        copytree(old_path, new_path)


def batch_aug(oriset, targetset, flip=False, rand_num=None,
              channel_num=None, zoom_num=None,
              shear_num=None, rotate_num=None):
    if flip == True:
        batch_image_process(oriset, targetset, process='flip')
    if shear_num is not None:
        for l in range(shear_num):
            batch_image_process(oriset, targetset, process='shear', stage=l)
    if zoom_num is not None:
        for x in range(zoom_num):
            batch_image_process(oriset, targetset, process='zoom', stage=x)
    if rand_num is not None:
        for i in range(rand_num):
            batch_image_process(oriset,targetset, process='random', stage=i)
    if channel_num is not None:
        for j in range(channel_num):
            batch_image_process(oriset, targetset, process='channel', stage=j)
    if rotate_num is not None:
        for y in range(rotate_num):
            batch_image_process(oriset, targetset, process='rotate')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", default=True, type=bool)
    args = parser.parse_args()
    ifcrop = args.crop
    rand_num = 11
    channel_num = 2
    zoom_num = 2
    shear_num = 2
    rotate_num = 2
    #batch_aug(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR, rand_num=1)
    #         zoom_num=zoom_num,
    #        shear_num=shear_num, rotate_num = rotate_num)
    batch_aug(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR, rand_num=rand_num,# channel_num=channel_num,
              zoom_num=zoom_num,
              shear_num=shear_num, rotate_num = rotate_num)
    #batch_aug(VALID_OLD_DATA_DIR, VALID_TARGET_DATA_DIR, rand_num=rand_num, channel_num=channel_num, zoom_num=zoom_num, shear_num=shear_num)

    copy_oriimg_to_new(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR)
    copy_oriimg_to_new(VALID_OLD_DATA_DIR, VALID_TARGET_DATA_DIR)
    #if ifcrop == True:
       # batch_image_process(TRAIN_TARGET_DATA_DIR, CROP_TRAIN_TARGET_DATA_DIR, process='crop')
      #  batch_image_process(VALID_TARGET_DATA_DIR, CROP_VALID_TARGET_DATA_DIR, process='crop')

    #batch_image_process(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR, process='crop')
    #batch_image_process(VALID_OLD_DATA_DIR, VALID_TARGET_DATA_DIR, process='crop')
    #batch_image_process(TEST_OLD_DATA_DIR, TEST_TARGET_DATA_DIR, process='crop')
