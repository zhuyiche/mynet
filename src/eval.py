import numpy as np
import scipy.misc as misc
import cv2
import os
import scipy.io as sio
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

TEST_DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
DATA_DIR = os.path.join(ROOT_DIR, 'aug')
def non_max_suppression(img, overlap_thresh=0.3, max_boxes=1200, r=8, prob_thresh=0.8):
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
    boxes = np.concatenate((x1s.reshape((x1s.shape[0], 1)), y1s.reshape((y1s.shape[0], 1)),
                            x2s.reshape((x2s.shape[0], 1)), y2s.reshape((y2s.shape[0], 1))), axis=1)
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
        # plt.imshow(gt_map)
        # plt.show()

        # pred_map = np.zeros((y_max, x_max), dtype='int')
        pred_map = np.zeros((500, 500)).astype('int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = min(result_map.sum(), gt.shape[0])

        precision = min(tp / (pred.shape[0] + epsilon), 1)
        recall = min(tp / (gt.shape[0] + epsilon), 1)
        f1_score = 2 * (precision * recall / (precision + recall + epsilon))
        gt_num = gt.shape[0]
        pred_num = pred.shape[0]
        return precision, recall, f1_score, tp, gt_num, pred_num


def mymetrics(model):
    import tensorflow as tf
    tp_num = 0
    gt_num = 0
    pred_num = 0

    for file in enumerate(os.listdir(os.path.join(TEST_DATA_DIR, 'test'))):
        #print(os.path.join(TEST_DATA_DIR, 'test'))
        #print(file[-1])
        file_path = os.path.join(TEST_DATA_DIR, 'test', str(file[-1]))
        #print(os.path.join(file_path, str(file[-1]) + '.bmp'))
        img = misc.imread(os.path.join(file_path, str(file[-1]) + '.bmp'))
        img = misc.imresize(img, (256, 256))
        img = img - 128.
        img = img / 128.
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        print(img.shape)
        result = model.predict(np.array(img))
        print('result.shape: {}'.format(result.shape))
        result = misc.imresize(result, (500, 500))
        result = result / 255.
        boxes = non_max_suppression(result)
        print('nms finises')
        gt = sio.loadmat(os.path.join(file_path, str(file) + 'detection.mat'))['detection']
        pred = []
        for i in range(boxes.shape[0]):
            x1 = boxes[i, 0]
            y1 = boxes[i, 1]
            x2 = boxes[i, 2]
            y2 = boxes[i, 3]
            cx = int(x1 + (x2 - x1) / 2)
            cy = int(y1 + (y2 - y1) / 2)
            pred.append([cx, cy])
        p, r, f1, tp = get_metrics(gt, pred)
        tp_num += tp
        gt_num += gt.shape[0]
        pred_num += np.array(pred).shape[0]
        print('good')
        precision = tp_num / (pred_num + epsilon)
        recall = tp_num / (gt_num + epsilon)
        f1_score = 2 * (precision * recall / (precision + recall + epsilon))

        return precision, recall, f1_score
