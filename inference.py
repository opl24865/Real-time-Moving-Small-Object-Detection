import os
import cv2
import torch
import numpy as np
import pandas as pd

from os.path import exists
from matplotlib import pyplot as plt
from tqdm import tqdm
from Dataloader_inference import BadmintonDataset_real
from torch.utils.data import DataLoader
from DetectionNet import DetectionNet
from collections import deque
from utils.util import *

# 初始化影片寫入器
output_video_path = "./output/demo/demo_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 MP4 編碼
fps = 25  # 設定 FPS（可依實際需求調整）
frame_size = (640, 480)  # 影像尺寸
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

def show_prediction_box(ground_img, keypoint_2d=None, conf=None):
    
    # 轉換 ground truth 圖像 [C, H, W] → [H, W, C]
    y = ground_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # 尺寸視 video resolution 調整
    ax.imshow(y)
    ax.axis('off')

    # 畫 bounding box 和 confidence 
    if keypoint_2d is not None :     
        x, y_kpt = keypoint_2d[0,0],keypoint_2d[0,1]
        c = conf
        box_size = 5
        rect = plt.Rectangle((x - box_size, y_kpt - box_size), 2 * box_size, 2 * box_size,
                          linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    # Matplotlib 圖片 → OpenCV 畫面
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_resized = cv2.resize(img_array, frame_size)

    video_writer.write(img_resized)
    plt.close(fig)

if __name__ == "__main__":
    
    dataset = BadmintonDataset_real(height=480, width=640)
    inf_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    mean_feature = np.load('mean_embedding.npy')  
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # create model
    B_model = DetectionNet()
    checkpoint_detect_dir = './exp/badv9_diffv10_ps_sigmoid/'
    checkpoint = torch.load(f'{checkpoint_detect_dir}/model_cur.pt')
    B_model.load_state_dict(checkpoint['model_state_dict'])
    
    B_model.to(device)
   
    B_model.eval()

    frame_buffer = deque(maxlen=2)
    keypoint_list = []
    keypoint_2d=[]
    ball = 0
    for index, images in enumerate(tqdm(inf_dataloader)):
        images = images.to(device)
        
        with torch.no_grad():
            y_pred = B_model(images)
            
        y_p = y_pred[:,2].detach().cpu().numpy()
        y_p = (y_p > 0.5).astype(np.uint8) * 255
        
        cx_pred, cy_pred = get_object_center(y_p[0])
        vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
        keypoint_2d.append((cx_pred,cy_pred))
        keypoint_list.append([index, ball , np.array(keypoint_2d[0])])
        show_prediction_box(images[:,2], np.array(keypoint_2d))
        keypoint_2d = []

    keypoint_array = np.array(keypoint_list, dtype=object)  
    x = [k[2][0] for k in keypoint_list]  # keypoint_2d[0, 0] 是 x
    y = [k[2][1] for k in keypoint_list]  # keypoint_2d[0, 1] 是 y
    index = [k[0] for k in keypoint_list]    # 幀索引
    ball = [k[1] for k in keypoint_list]
    key_dict = {'Frame':index, 'Visibility':ball, 'X': x, 'Y': y}
    
    df = pd.DataFrame(key_dict)
    df.to_csv('key.csv', index=False)
    video_writer.release()
    cv2.destroyAllWindows()