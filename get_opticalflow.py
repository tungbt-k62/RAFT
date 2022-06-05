import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, filename):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    cv2.imwrite("/content/flow224_5/Untitled{}_frame_{}.jpg".format(video_id, frame_id), flo)
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        f_path = args.txt_path
        f = open(f_path, "r")
        images = f.read().splitlines()
        f.close()
        # images = sorted(images)
        for id, item in enumerate(images):
#             if id < 8:
#                 continue
            path, speed, video_id, frame_id = item.split(" ")
            imfile1 = "/content/data1_6/Untitled{}_frame_{}.jpg".format(video_id, frame_id)
            imfile2 = "/content/data1_6/Untitled{}_frame_{}.jpg".format(video_id, frame_id-4)
            print(imfile2, imfile1)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            viz(image1, flow_up, video_id, frame_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--txt_path', help='txt to path img')
    args = parser.parse_args()

    demo(args)
