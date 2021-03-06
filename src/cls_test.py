from encoder_decoder_object_det import Detnet
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config
#from metric import non_max_suppression, get_metrics
import scipy.io as sio
import cv2
from encoder_append import *
from eval import mymetrics
from deeplabv3_plus import Deeplab

weight_path = 'multi_fcn36_loss:fd_det:0.13_fkg:0.5_bkg:0.5_lr:0.01_train.h5'
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

WEIGHT_DIR = os.path.join(ROOT_DIR, 'model_weights')
IMG_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det', 'train')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det', 'test')
#IMG_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det', 'test')
epsilon = 1e-6

def non_max_suppression(img, overlap_thresh=0.3, max_boxes=1200, r=8, prob_thresh=0.6):  # net_4_w6_di2.pkl
    # over=0.2, max=1200,r=7,prob=0.85 --> P:0.837 R:0.894 F:0.865
    # over=0.3, max=1200,r=8,prob=0.85 --> P:0.824 R:0.920 F:0.869
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    probs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < prob_thresh:
                img[i, j] = 0
            else:
                x1 = max(j - r, 0)
                y1 = max(i - r, 0)
                x2 = min(j + r, img.shape[1] - 1)
                y2 = min(i + r, img.shape[0] - 1)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                probs.append(img[i, j])
    x1s = np.array(x1s)
    y1s = np.array(y1s)
    x2s = np.array(x2s)
    y2s = np.array(y2s)
    # print(x1s.shape)
    boxes = np.concatenate((x1s.reshape((x1s.shape[0], 1)), y1s.reshape((y1s.shape[0], 1)), x2s.reshape((x2s.shape[0], 1)),
                            y2s.reshape((y2s.shape[0], 1))), axis=1)
    # print(boxes.shape)
    probs = np.array(probs)
    pick = []
    area = (x2s - x1s) * (y2s - y1s)
    indexes = np.argsort([i for i in probs])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1_int = np.maximum(x1s[i], x1s[indexes[:last]])
        yy1_int = np.maximum(y1s[i], y1s[indexes[:last]])
        xx2_int = np.minimum(x2s[i], x2s[indexes[:last]])
        yy2_int = np.minimum(y2s[i], y2s[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
            # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    # print(boxes.shape)
    return boxes


def get_metrics(gt, pred, r=6):
    # calculate precise, recall and f1 score
    gt = np.array(gt).astype('int')
    if pred == []:
        if gt.shape[0] == 0:
            return 1, 1, 1, 0
        else:
            return 0, 0, 0, 0


    pred = np.array(pred).astype('int')

    temp = np.concatenate([gt, pred])

    if temp.shape[0] != 0:
        x_max = np.max(temp[:, 0]) + 1
        y_max = np.max(temp[:, 1]) + 1

        #gt_map = np.zeros((y_max, x_max), dtype='int')
        gt_map = np.zeros((500, 500)).astype(np.int8)
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(x_max, x + r)
            y2 = min(y_max, y + r)
            # gt_map[y1:y2,x1:x2] = 1
            cv2.circle(gt_map, (x, y), r, 1, -1)
        #plt.imshow(gt_map)
        #plt.show()

        #pred_map = np.zeros((y_max, x_max), dtype='int')
        pred_map = np.zeros((500,500),dtype='int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = result_map.sum()

        precision = min(tp / (pred.shape[0] + epsilon),1)
        recall = min(tp / (gt.shape[0] + epsilon),1)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        gt_num = gt.shape[0]
        pred_num = pred.shape[0]
        return precision, recall, f1_score, tp, gt_num, pred_num


def eval_single_img(model, img_dir, print_img=True,
                    prob_threshold=None, print_single_result=True):
    image_path = os.path.join(IMG_DIR, img_dir, img_dir+ '.bmp')
    img = misc.imread(image_path)
    img = misc.imresize(img, (256, 256))#, interp='nearest')
    img = img - 128.0
    img = img / 128.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    output = model.predict(img)[0]
    output1 = output[:, :, 1]
    output2 = output[:, :, 2]
    output3 = output[:, :, 3]
    output4 = output[:, :, 4]

    def _predic_crop_image(img, print_img=print_img):
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        output = model.predict(img)[0]
        output = np.argmax(output)
        return output
    argma = np.argmax(output)
    #print(argma.shape)
    if print_img:
        plt.imshow(output[output[:, :, 1] > 0.3])
        plt.title(weight_path)
        plt.colorbar()
        plt.show()
    #print(output.shape)

    epi_p, epi_r, epi_f1, epi_tp, epi_gt, epi_prednum = cls_score_single_img(output4, img_dir=img_dir, type='epi',
                                                                             prob_threshold=prob_threshold,
                                                                             print_single_result=print_single_result)
    fib_p, fib_r, fib_f1, fib_tp, fib_gt, fib_prednum = cls_score_single_img(output2, img_dir=img_dir,
                                                                             prob_threshold=prob_threshold, type='fib',
                                                                             print_single_result=print_single_result)
    inf_p, inf_r, inf_f1, inf_tp, inf_gt, inf_prednum = cls_score_single_img(output3, img_dir=img_dir,
                                                                             prob_threshold=prob_threshold, type='inf',
                                                                             print_single_result=print_single_result)
    other_p, other_r, other_f1, other_tp, other_gt, other_prednum = cls_score_single_img(output4, img_dir=img_dir,
                                                                             prob_threshold=prob_threshold, type='other',
                                                                             print_single_result=print_single_result)
    print('epi_p: {}, epi_r: {}, epi_f1: {}, fib_f1: {}, inf_f1: {}, other_f1: {}'.format(epi_p, epi_r, epi_f1, fib_f1, inf_f1, other_f1))
    total_gt = epi_gt + fib_gt + inf_gt + other_gt
    total_tp = epi_tp + fib_tp + inf_tp + inf_tp
    total_prednum = epi_prednum + fib_prednum + inf_prednum + inf_prednum
    avg_p = (epi_p + fib_p + inf_p + other_p)/4
    avg_r = (epi_r + fib_r + inf_r + other_r)/4
    avg_f1 = (epi_f1 + fib_f1 + inf_f1 + other_f1)/4
    return avg_p, avg_r, avg_f1, total_tp, total_gt, total_prednum


def cls_score_single_img(input, img_dir, type, prob_threshold=None, print_single_result=True):
    input = misc.imresize(input, (500, 500))
    input = input / 255.
    boxes = non_max_suppression(input, prob_thresh=prob_threshold)

    num_of_nuclei = boxes.shape[0]

    epi_path = os.path.join(IMG_DIR, img_dir, img_dir + '_epithelial.mat')
    fib_path = os.path.join(IMG_DIR, img_dir, img_dir + '_fibroblast.mat')
    inf_path = os.path.join(IMG_DIR, img_dir, img_dir + '_inflammatory.mat')
    other_path = os.path.join(IMG_DIR, img_dir, img_dir + '_others.mat')

    epi_gt = sio.loadmat(epi_path)['detection']
    fib_gt = sio.loadmat(fib_path)['detection']
    inf_gt = sio.loadmat(inf_path)['detection']
    other_gt = sio.loadmat(other_path)['detection']

    if type == 'epi':
        gt = epi_gt
    elif type == 'fib':
        gt = fib_gt
    elif type == 'inf':
        gt = inf_gt
    elif type == 'other':
        gt = other_gt

    outputbase = cv2.imread(os.path.join(IMG_DIR, img_dir, img_dir + '.bmp'))
    if print_single_result:
        print('----------------------------------')
        print('This is {}'.format(img_dir))
        print('detected: {}, ground truth: {}'.format(num_of_nuclei, gt.shape[0]))
    pred = []
    for i in range(boxes.shape[0]):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        # cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
        cv2.circle(outputbase, (cx, cy), 3, (255, 255, 0), -1)
        pred.append([cx, cy])
    p, r, f1, tp, aaa, bbb = get_metrics(gt, pred)
    return p, r, f1, tp, aaa, bbb


def eval_testset(model, prob_threshold=None, print_img=False, print_single_result=True):
    total_p, total_r, total_f1, total_tp = 0, 0, 0, 0
    tp_total_num, gt_total_num, pred_total_num = 0, 0, 0
    for img_dir in os.listdir(IMG_DIR):
        p, r, f1, tp, gt, pred = eval_single_img(model, img_dir, print_img=print_img,
                                       print_single_result=print_single_result,
                                       prob_threshold=prob_threshold)
        #print('{} p: {}, r: {}, f1: {}, tp: {}'.format(img_dir, p, r, f1, tp))
        total_p += p
        total_r += r
        total_f1 += f1
        total_tp += tp

        tp_total_num += tp
        gt_total_num += gt
        pred_total_num += pred

    precision = tp_total_num/(pred_total_num + epsilon)
    recall = tp_total_num / (gt_total_num + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    #print('Over points, the precision: {}, recall: {}, f1: {}'.format(precision, recall, f1_score))
   # if prob_threshold is not None:
      #  print('The nms threshold is {}'.format(prob_threshold))
    #print('Over test set, the average P: {}, R: {}, F1: {}, TP: {}'.format(total_p/20,total_r/20,total_f1/20,total_tp/20))
    return total_p/20, total_r/20, total_f1/20

def eval_weights_testset(weightsdir):
    weights_dict = {'best_p': 0, 'best_r': 0, 'best_f1': 0,
                    'best_p_model': None, 'best_r_mode': None,
                    'best_f1_model':None, 'best_p_prob': None,
                    'best_r_prob': None, 'best_f1_prob': None}
    print('Start model evalutation')
    for i, weight_dir in enumerate(os.listdir(weightsdir)):
        if 'resnet50' in weight_dir:
            model = Detnet().detnet_resnet50_backbone()
        elif 'resnet101' in weight_dir:
            model = Detnet().detnet_resnet101_backbone()
        elif 'resnet152' in weight_dir:
            model = Detnet().detnet_resnet152_backbone()

        model.load_weights(os.path.join(WEIGHT_DIR, weight_dir))
        for prob in prob_threshhold:
            print('###################')
            print('current model is {} at threshold {}'.format(weight_dir, prob))
            avg_p, avg_r, avg_f1 = eval_testset(model, prob_threshold=prob, print_img=False,
                                                print_single_result=False)
            if np.round(avg_r) < 0.8:
                break
            if weights_dict['best_p'] == 0:
                weights_dict['best_p'], weights_dict['best_r'], weights_dict['best_f1'] = avg_p, avg_r, avg_f1
            else:
                if avg_p > weights_dict['best_p'] and avg_r > 0.6 and avg_f1 > 0.6:
                    weights_dict['best_p'] = avg_p
                    weights_dict['best_p_model'] = weight_dir
                    weights_dict['best_p_prob'] = prob
                if avg_r > weights_dict['best_r'] and avg_p > 0.6 and avg_f1 > 0.6:
                    weights_dict['best_r'] = avg_r
                    weights_dict['best_r_model'] = weight_dir
                    weights_dict['best_r_prob'] = prob
                if avg_f1 > weights_dict['best_f1'] and avg_r > 0.6 and avg_p > 0.6:
                    weights_dict['best_f1'] = avg_f1
                    weights_dict['best_f1_model'] = weight_dir
                    weights_dict['best_f1_prob'] = prob

        print('best precision is {} with model {} at threshold {}'.format(weights_dict['best_p'],
                                                                          weights_dict['best_p_model'],
                                                                          weights_dict['best_p_prob']))
        print('best recall is {} with model {} at threshold {}'.format(weights_dict['best_r'],
                                                                       weights_dict['best_p_model'],
                                                                       weights_dict['best_p_prob']))
        print('best f1 is {} with model {} at threshold {}'.format(weights_dict['best_f1'],
                                                                   weights_dict['best_f1_model'],
                                                                   weights_dict['best_f1_prob']))


def test_11(model):
    p, r, f1, tp, gt, pred= eval_single_img(model, 'img11', print_img=True,
                                   print_single_result=False,
                                   prob_threshold=0.8)
    print('Over test set, the average P: {}, R: {}, F1: {}, TP: {}'.format(p, r, f1, tp))


def MyMetrics(model, weight):
    def _get_metrics(gt, pred, r=6):
        # calculate precise, recall and f1 score
        gt = np.array(gt).astype('int')
        if pred == []:
            if gt.shape[0] == 0:
                return 1, 1, 1, 0
            else:
                return 0, 0, 0, 0

        pred = np.array(pred).astype('int')

        temp = np.concatenate([gt, pred])

        if temp.shape[0] != 0:
            x_max = np.max(temp[:, 0]) + 1
            y_max = np.max(temp[:, 1]) + 1

            # gt_map = np.zeros((y_max, x_max), dtype='int')
            gt_map = np.zeros((500, 500), dtype='int')
            for i in range(gt.shape[0]):
                x = gt[i, 0]
                y = gt[i, 1]
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(x_max, x + r)
                y2 = min(y_max, y + r)
                gt_map[y1:y2, x1:x2] = 1
                # cv2.circle(gt_map, (x, y), r, 1, -1)
            # pred_map = np.zeros((y_max, x_max), dtype='int')
            pred_map = np.zeros((500, 500), dtype='int')
            for i in range(pred.shape[0]):
                x = pred[i, 0]
                y = pred[i, 1]
                pred_map[y, x] = 1

            result_map = gt_map * pred_map
            tp = result_map.sum()

            precision = tp / (pred.shape[0] + epsilon)
            recall = tp / (gt.shape[0] + epsilon)
            f1_score = 2 * (precision * recall / (precision + recall + epsilon))

            return precision, recall, f1_score, tp
    model.load_weights(weight)
    path = TEST_IMG_DIR
    tp_num = 0
    gt_num = 0
    pred_num = 0
    precision = 0
    recall = 0
    f1_score = 0
    type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    for i in range(81, 101):
        file_name = os.path.join(path, 'img' + str(i))
        filename = os.path.join(file_name, 'img' + str(i) + '.bmp')

        if os.path.exists(filename):
            gtpath = './CRCHistoPhenotypes_2016_04_28/Classification'
            imgname = 'img' + str(i)
            img = misc.imread(filename)
            img = misc.imresize(img, (256, 256))
            img = img - 128.
            img = img / 128.
            print(img.shape)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            print(img.shape)
            #img = np.transpose(img, (0, 3, 1, 2))
            #img = torch.Tensor(img).cuda()
            result = model.predict(img)
            #result = result.cpu().detach().numpy()
            result = result[0]
            print(result.shape)
            #result = np.exp(result)
            for j in range(1, 5):
                result_tmp = result[:, :, j]
                result_tmp = misc.imresize(result_tmp, (500, 500))
                result_tmp = result_tmp / 255.
                boxes = non_max_suppression(result_tmp, prob_thresh=0.2)
                matname = imgname + '_' + type_group[j-1] + '.mat'
                matpath = os.path.join(file_name, matname)
                gt = sio.loadmat(matpath)['detection']
                pred = []
                for k in range(boxes.shape[0]):
                    x1 = boxes[k, 0]
                    y1 = boxes[k, 1]
                    x2 = boxes[k, 2]
                    y2 = boxes[k, 3]
                    cx = int(x1 + (x2 - x1) / 2)
                    cy = int(y1 + (y2 - y1) / 2)
                    pred.append([cx, cy])
                p, r, f1, tp = _get_metrics(gt, pred)
                print('img{} f1: {}'.format(str(i), f1) )
                tp_num += tp
                gt_num += gt.shape[0]
                pred_num += np.array(pred).shape[0]
    precision = tp_num / (pred_num + epsilon)
    recall = tp_num / (gt_num + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return precision, recall, f1_score

if __name__ == '__main__':
    model = Fcn_det().relufirst_fcn36_deconv_backbone()
    for weight in os.listdir(WEIGHT_DIR):
        print(weight)
        weightp = os.path.join(WEIGHT_DIR, weight)
        p, r, f1 = MyMetrics(model, weightp)
        print('total p: {}, total r: {}, total f1: {}'.format(p, r, f1))
    """
    imgdir = 'img' + str(2)

    model = Fcn_det().relufirst_fcn36_deconv_backbone()
    image_dir = '/home/zhuyiche/Desktop/cell/CRCHistoPhenotypes_2016_04_28/cls_and_det/train/img1'
    #model.load_weights()
    #eval_weights_testset(WEIGHT_DIR)
    for weight in os.listdir(WEIGHT_DIR):
        print(weight)
        weightp = os.path.join(WEIGHT_DIR, weight)
        model.load_weights(weightp)
    #model.load_weights(os.path.join(WEIGHT_DIR, weight_path))
        prob_threshhold = [0.2]#0.3, 0.4,0.43, 0.45, 0.48, 0.5,0.52,0.55,0.58, 0.60,0.62,0.65, 0.7, 0.8, 0.9]
        #eval_single_img(model, imgdir)
        for prob in prob_threshhold:
            print('The nms threshold is ', prob)
            eval_testset(model, prob_threshold=prob, print_img=False, print_single_result=False)

"""







'''
    print(p, r, f1)
    tp_num += tp
    gt_num += gt.shape[0]
    pred_num += np.array(pred).shape[0]
precision = tp_num / (pred_num + epsilon)
recall = tp_num / (gt_num + epsilon)
f1_score = 2 * (precision * recall / (precision + recall + epsilon))

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
'''

#model = Detnet(0.25).detnet_backbone()
#model.load_weights('../model_weights/focal_aug_elas_trans_smooth_4_bkg_1_train.h5')
#img = misc.imread('img90.bmp')

