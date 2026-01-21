import os
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from math import ceil
from collections import deque
from tqdm import tqdm

class BadmintonDataset_real(Dataset):
    def __init__(self, height=480, width=640):
        self.root_dir = './video'
        self.height = height
        self.width = width
        self.total_data = self._load_data()
        self.total_length = len(self.total_data)

    def _load_data(self):
        img_list = []
        frame_buffer = deque(maxlen=3)
        video_file = os.listdir(self.root_dir)
        for i in video_file:
            if i == '.ipynb_checkpoints':
                continue
            video_path = os.path.join(self.root_dir,i)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
            with tqdm(total=total_frames, desc="處理中") as pbar:
                while True:
                    ret,frame = cap.read()
                    if not ret:
                        print("影片處理完成")   # 如果讀取錯誤，印出訊息
                        break
                    pbar.update(1)
                    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                        print("跳過無效幀")
                        continue
                    frame_buffer.append(frame)
                    if len(frame_buffer) == 3:
                        frame1 = np.expand_dims(frame_buffer[0],axis=0)
                        frame2 = np.expand_dims(frame_buffer[1],axis=0)
                        frame3 = np.expand_dims(frame_buffer[2],axis=0)
                        imgstack = np.vstack((frame1,frame2,frame3))
                        img_list.append(imgstack)
                    else:
                        continue
                        
            cap.release()
        return img_list

    def _preprocessed(self, frame1, frame2, frame3):
        trans = transforms.Compose([transforms.ToTensor()])
        resized_imgs = []
        resized_img1 = cv2.resize(frame1, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized_img2 = cv2.resize(frame2, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized_img3 = cv2.resize(frame3, (self.width, self.height), interpolation=cv2.INTER_AREA)

        resized_imgs.append(resized_img1)
        resized_imgs.append(resized_img2)
        resized_imgs.append(resized_img3)

        resized_imgs = [trans(img) for img in resized_imgs]

        return resized_imgs[0], resized_imgs[1], resized_imgs[2]
        
    def __len__(self):
        total_length = self.total_length
        return total_length

    def __getitem__(self, index):
        img1, img2, img3 = self.total_data[index]
        resized_img1, resized_img2, resized_img3 = self._preprocessed(img1,img2,img3)
        stacked_images = torch.stack((resized_img1, resized_img2, resized_img3))
        return stacked_images




