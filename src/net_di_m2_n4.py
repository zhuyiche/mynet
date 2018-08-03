import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
import dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import scipy.io as sio

epsilon = 1e-7


class identity_block(nn.Module):
    '''(Conv=>BN=>ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(identity_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y

class identity_block_di(nn.Module):
    '''(Conv=>BN=>ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(identity_block_di, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y


class strided_block(nn.Module):
    '''downsample featuremap between modules'''

    def __init__(self, in_ch, out_ch):
        super(strided_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y


class conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Res_Module, self).__init__()
        if downsample:
            self.conv1 = strided_block(in_ch, out_ch)
        else:
            self.conv1 = identity_block(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Res_Module_di(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Res_Module_di, self).__init__()
        if downsample:
            self.conv1 = strided_block(in_ch, out_ch)
        else:
            self.conv1 = identity_block(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            identity_block(out_ch, out_ch),
            identity_block_di(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block_di(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block_di(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block_di(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Net_4_di(nn.Module):
    def __init__(self, log_softmax=True):
        super(Net_4_di, self).__init__()
        self.log_softmax = log_softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = conv_1x1(64, 2)
        self.conv3 = conv_1x1(128, 2)
        self.conv4 = conv_1x1(256, 2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module_di(64, 128, downsample=True)
        self.Module4 = Res_Module_di(128, 256, downsample=True)
        self.conv6 = nn.Conv2d(6, 2, 3, padding=1)
        self.out = nn.LogSoftmax(dim=1)
        self.conv0 = conv_1x1(32, 2)
        self.conv5 = conv_1x1(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        y0 = self.conv0(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        x = self.Module4(x)
        y3 = self.conv4(x)
        y3 = self.deconv1(y3)
        c1 = torch.cat([y2, y3], dim=1)
        c1 = self.deconv2(c1)
        y3 = self.deconv1(y3)
        y3 = self.deconv4(y3)
        out_4 = y3
        c2 = torch.cat([y1, c1], dim=1)
        c2 = self.deconv3(c2)
        c3 = torch.cat([y0, c2], dim=1)
        c3 = self.conv5(c3)
        c1 = self.deconv4(c1)
        out_2 = c1
        out = c3
        o = torch.cat([out, out_2, out_4], dim=1)
        final_out = self.conv6(o)
        final_out = self.out(final_out)
        return final_out

def non_max_suppression(img, overlap_thresh=0.1, max_boxes=1200, r=5, prob_thresh=0.85):
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


def get_metrics(gt, pred, r=3):
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

        gt_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(x_max, x + r)
            y2 = min(y_max, y + r)
            gt_map[y1:y2, x1:x2] = 1

        pred_map = np.zeros((y_max, x_max), dtype='int')
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


def MyMetrics(model):
    path = './CRCHisto'
    tp_num = 0
    gt_num = 0
    pred_num = 0
    precision = 0
    recall = 0
    f1_score = 0

    for i in range(81, 101):
        filename = os.path.join(path, 'img' + str(i) + '.bmp')
        if os.path.exists(filename):
            gtpath = './CRCHistoPhenotypes_2016_04_28/Detection'
            imgname = 'img' + str(i)
            img = misc.imread(filename)
            img = misc.imresize(img, (256, 256))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.Tensor(img).cuda()
            result = model(img)
            result = result.cpu().detach().numpy()
            result = np.transpose(result, (0, 2, 3, 1))[0]
            result = np.exp(result)
            result = result[:, :, 1]
            result = misc.imresize(result, (500, 500))
            result = result / 255.
            boxes = non_max_suppression(result)
            matname = imgname + '_detection.mat'
            matpath = os.path.join(gtpath, imgname, matname)
            gt = sio.loadmat(matpath)['detection']
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
    precision = tp_num / (pred_num + epsilon)
    recall = tp_num / (gt_num + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return precision, recall, f1_score

def train_2(model, weight=None, data_dir='', preprocess=True, gpu=True, batch_size=2, num_epochs=100):
    if weight == None:
        weight = torch.Tensor([1, 1])
    else:
        weightid = (str(weight[1])).split('.')[-1]
        weight = torch.Tensor(weight)


    writer = SummaryWriter()

    data = dataset.CRC_softmask(data_dir, target_size=256)
    x_train, y_train = data.load_train(preprocess=preprocess)
    train_count = len(x_train)

    x_val, y_val = data.load_val(preprocess=preprocess)
    val_count = len(x_val)
    val_steps = int(val_count / batch_size)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val], axis=1)
    valset = torch.Tensor(valset)

    if gpu:
        model = model.cuda()
        trainset = trainset.cuda()
        valset = valset.cuda()
        weight = weight.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss = nn.NLLLoss(weight=weight)
    # NLLLoss = MyLoss(weight=weight)
    best_loss = 99999.0
    best_f1 = 0.0

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3, :, :]
            train_masks = datapack[:, 3:, :, :]

            train_masks = train_masks.long()

            train_masks = train_masks.view(
                train_masks.size()[0],
                train_masks.size()[2],
                train_masks.size()[3]
            )

            optimizer.zero_grad()
            train_out = model(train_imgs)
            t_loss = NLLLoss(train_out, train_masks)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                writer.add_scalar('train_loss', train_loss / 10, (105 * epoch + i + 1))
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3, :, :]
            val_masks = datapack[:, 3:, :, :]

            val_masks = val_masks.long()
            val_masks = val_masks.view(
                val_masks.size()[0],
                val_masks.size()[2],
                val_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_out = model(val_imgs)
            v_loss = NLLLoss(val_out, val_masks)
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                '''
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_name = './ckpt/net_4_w' + weightid + '_di.pkl'
                    torch.save(model.state_dict(), save_name)
                '''
                end = time.time()
                time_spent = end - start
                writer.add_scalar('val_loss', val_loss, epoch)
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss = 0.0
                p, r, f = MyMetrics(model)
                if f > best_f1:
                    best_f1 = f
                    save_name = './ckpt/net_di_m2_n4_w' + weightid + '.pkl'
                    torch.save(model.state_dict(), save_name)
                writer.add_scalar('precision', p, epoch)
                writer.add_scalar('recall', r, epoch)
                writer.add_scalar('f1_score', f, epoch)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                print('******************************************************************************')

    # writer.export_scalars_to_json('./loss.json')
    writer.close()


# net = Net()
# train(net, weight=[0.1, 2], data_dir='./aug',num_epochs=300)
if __name__ == '__main__':
    net = Net_4_di()
    train_2(net, weight=[0.1, 1.0], data_dir='./aug', num_epochs=200, batch_size=4)
    train_2(net, weight=[0.1, 1.8], data_dir='./aug', num_epochs=200, batch_size=4)
    train_2(net, weight=[0.1, 1.2], data_dir='./aug', num_epochs=200, batch_size=4)
    train_2(net, weight=[0.1, 1.6], data_dir='./aug', num_epochs=200, batch_size=4)
    train_2(net, weight=[0.1, 1.4], data_dir='./aug', num_epochs=200, batch_size=4)