import os
import torch
import time
import argparse
import torchvision.transforms as transforms
import signal
import warnings

from DetectionNet import DetectionNet
from Dataloader_train import BadmintonDataset_real
from torch.utils.data import DataLoader
from trainer import *
from utils.util import WarmupLinearSchedule, WarmupCosineSchedule, signal_handler
warnings.filterwarnings("ignore")
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='DiffNetTrack')
parser.add_argument('--num_frame', type=int, default=2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--data_dir', type=str, default='./Processed_dataset')
parser.add_argument('--save_dir', type=str, default='./exp')
parser.add_argument('--resume_training', action='store_true', default=False)
parser.add_argument('--tolerance', type=float, default=4)
parser.add_argument('--train_schedule', type=str, default='cosine')

args = parser.parse_args()

model_name = args.model_name
num_frame = args.num_frame
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
data_dir = args.data_dir
save_dir = args.save_dir
resume_training = args.resume_training
tolerance = args.tolerance
display_step = 1000
train_schedule = args.train_schedule


transform = transforms.Compose([transforms.ToTensor()])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load dataset
train_dataset = BadmintonDataset_real(transform=transform, height=480, width=640, size=15, variance=15, if_variance=True)
test_dataset = BadmintonDataset_real(transform=transform, train_data=False, height=480, width=640, size=15, variance=15, if_variance=True)
train_loader = DataLoader(train_dataset, batch_size=12, num_workers=8, shuffle=True, pin_memory=True,persistent_workers=True,prefetch_factor=4)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=3, shuffle=False)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create model and optimizer
    Badminton_model = DetectionNet().to(device)
    optimizer = torch.optim.Adam(Badminton_model.parameters(), lr=learning_rate)

    # 設定learning rate scheduler(default使用cosine)
    t_total = 210700
    if train_schedule == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=21070, t_total=t_total)
        # scheduler.last_epoch = 109830
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=21070, t_total=t_total)

    # 是否載入之前訓練好的權重   
    if resume_training:
        checkpoint = torch.load(f'{save_dir}/badv9_diffv10_ps_sigmoid/model_cur_11.pt', map_location=device)
        Badminton_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_list = checkpoint['loss_list']
        accuracy= checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume training from epoch {start_epoch}.')
    else:
        start_epoch = 0
        accuracy = []
        loss_list = []
    
    # training loop
    train_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        loss = train_badmintonNet(epoch, Badminton_model, optimizer, scheduler, train_loader, WeightedBinaryCrossEntropy, display_step, device)
        cur_accuracy, precision, recall, TP, TN, FP1, FP2, FN = evaluation(Badminton_model, test_loader, epoch, tolerance, display_step, device)
        
        torch.save(dict(epoch=epoch,
                        model_state_dict=Badminton_model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        loss_list=loss_list,
                        accuracy=accuracy
                        ), f'{save_dir}/model_cur.pt')
        print("accuracy",accuracy)
        print("cur_accuracy",cur_accuracy)
        if cur_accuracy > max(accuracy, default=float('-inf')):
            torch.save(dict(epoch=epoch,
                        model_state_dict=Badminton_model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        loss_list=loss_list,
                        accuracy=accuracy
                        ), f'{os.path.join(save_dir,"best_model")}/best_model.pt')
                
            with open(f'{os.path.join(save_dir,"best_model")}/best_model.txt', "w", encoding="utf-8") as file:
                file.write(f'epoch: {epoch}, train_loss: {loss}, test_accuracy: {cur_accuracy}, precision:{precision}, recall:{recall}, TP:{len(TP)}, TN:{len(TN)}, FP1:{len(FP1)}, FP2:{len(FP2)}, FN:{len(FN)}')

        accuracy.append(cur_accuracy)
        print(loss)
        loss_list.append(loss)
        
    print(f'runtime: {(time.time() - train_start_time) / 3600.:.2f} hrs')
    print('Done......')
