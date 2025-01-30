import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from lib.pvt import PolypPVT
from DuAT import DuAT
from utils.dataloader import test_dataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='checkpoints/DuAT-best.pth')
    opt = parser.parse_args()
    model = DuAT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # ##### save_path #####
    save_path = 'result_map/DuAT/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = 'archive/ISIC2018_Task1-2_Training_Input/'
    gt_root = 'archive/ISIC2018_Task1_Training_GroundTruth/'
    num1 = len(os.listdir(gt_root)) - 2
    test_loader = test_dataset(image_root, gt_root, 352)

    for i in tqdm(range(num1)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        P1, P2 = model(image)
        res = F.upsample(P1 + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
    print('Finish!')
