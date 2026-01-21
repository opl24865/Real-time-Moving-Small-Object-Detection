from torch.optim.lr_scheduler import LambdaLR
import math
import torch
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, imsave
import sys
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def euclidean_loss(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()
    # return torch.sum((pred - target) ** 2, dim=1).mean()
def WeightedBinaryCrossEntropy(y, y_pred):
    # epsilon = 1e-7
    # pred_ball_exist = pred_ball_exist.squeeze()
    # existence_loss = F.binary_cross_entropy(pred_ball_exist, target_ball_existence.float())
    # pred_ball_exist = pred_ball_exist.unsqueeze(2).unsqueeze(3)
    # y_pred = y_pred * pred_ball_exist
    # WBCE = (-1)*(torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    gamma = 1.5
    loss = (-1)*(torch.square(1 - y_pred) * (torch.clamp(1 - y_pred, 1e-7, 1)** gamma) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) + torch.square(y_pred)* ((torch.clamp(y_pred, 1e-7, 1)) ** gamma) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    return torch.mean(loss) 

def show_prediction_train(y_pred, ground_img, epoch, step):
    y_pred_np = y_pred.view(-1, 480, 640).detach().cpu().numpy().transpose(1, 2, 0)
    ground_img_np = ground_img.view(-1, 3, 480, 640)[0].detach().cpu().numpy().transpose(1, 2, 0)
    savepath = os.path.join("./output/train", f'{epoch}')
    os.makedirs(savepath, exist_ok = True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 顯示地面真值彩色圖像
    axes[0].imshow(ground_img_np)
    axes[0].axis('off')
    axes[0].set_title("Ground Truth Image")

    # 顯示地面真值圖像，並疊加 heatmap
    axes[1].imshow(ground_img_np)
    axes[1].imshow(y_pred_np[:, :, 0], cmap='jet', alpha=0.5)  # 疊加 heatmap
    axes[1].axis('off')
    axes[1].set_title("Prediction Heatmap")

    # 儲存結果圖片
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'{step}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def show_prediction_test(y_pred, ground_img, epoch, step):
    y_pred_np = y_pred.detach().cpu().numpy()
    # print(y_pred_np.shape)
    y = ground_img.detach().cpu().numpy().transpose(1,2,0)
    savepath = os.path.join("./output/test", f'{epoch}')
    os.makedirs(savepath, exist_ok = True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 顯示地面真值彩色圖像
    axes[0].imshow(y)
    axes[0].axis('off')
    axes[0].set_title("Ground Truth Image")

    # 顯示地面真值圖像，並疊加 heatmap
    axes[1].imshow(y)
    axes[1].imshow(y_pred_np, cmap='jet', alpha=0.5)  # 疊加 heatmap
    axes[1].axis('off')
    axes[1].set_title("Prediction Heatmap")

    # 儲存結果圖片
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'{step}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def get_confusion_matrix(y_pred, y_true, y_coor, index, tolerance):
    """ Helper function Generate input sequences from frames.

        args:
            indices - A tf.EagerTensor of indices for sequences
            y_pred - A tf.EagerTensor of predicted heatmap sequences
            y_true - A tf.EagerTensor of ground-truth heatmap sequences
            y_coor - A tf.EagerTensor of ground-truth coordinate sequences
            tolerance - A int speicfying the tolerance for FP1
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)
        returns:
            TP, TN, FP1, FP2, FN - Lists of tuples of all the prediction results
                                    each tuple specifying (sequence_id, frame_no)
    """
    TP, TN, FP1, FP2, FN = [], [], [], [], []
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()

    for n in range(y_pred.shape[0]):
        num_frame = y_pred.shape[1]
        for f in range(num_frame):
            y_p = y_pred[n][f]
            y_t = y_true[n][f]
            c_t = y_coor[n][f]
            if np.amax(y_p) == 0 and np.amax(y_t) == 0:
                # True Negative: prediction is no ball, and ground truth is no ball
                TN.append((int(index), int(f)))
            elif np.amax(y_p) > 0 and np.amax(y_t) == 0:
                # False Positive 2: prediction is ball existing, but ground truth is no ball
                FP2.append((int(index), int(f)))
            elif np.amax(y_p) == 0 and np.amax(y_t) > 0:
                # False Negative: prediction is no ball, but ground truth is ball existing
                FN.append((int(index), int(f)))
            elif np.amax(y_p) > 0 and np.amax(y_t) > 0:
                # both prediction and ground truth are ball existing
                h_pred = y_p * 255
                h_true = y_t * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                cx_pred, cy_pred = int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)
                cx_true, cy_true = int(c_t[0]), int(c_t[1])
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                # print(f'dist:{dist}, cx_true:{cx_true}, cx_pred:{cx_pred}')
                if dist > tolerance:
                    # False Positive 1: prediction is ball existing, but is too far from ground truth
                    FP1.append((int(index), int(f)))
                else:
                    # True Positive
                    TP.append((int(index), int(f)))
    return TP, TN, FP1, FP2, FN

def get_metric(TP, TN, FP1, FP2, FN):
    """ Helper function Generate input sequences from frames.

        args:
            TP, TN, FP1, FP2, FN - Each float specifying the count for each result type of prediction

        returns:
            accuracy, precision, recall - Each float specifying the value of metric
    """
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return accuracy, precision, recall


# 偵測ctrl + c 中斷指令，並且釋放記憶體資源
def signal_handler(sig, frame):
    print('Exiting and cleaning up...')
    torch.cuda.empty_cache()
    sys.exit(0)


def get_object_center(heatmap):
    """ Get coordinates from the heatmap.

        args:
            heatmap - A numpy.ndarray of a single heatmap with shape (H, W)

        returns:
            ints specifying center coordinates of object
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]
    
    return int((target[0] + target[2] / 2)), int((target[1] + target[3] / 2))
