import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize

if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)
    model = model.cuda()

    img_path = '/content/drive/MyDrive/MaskGAN_demo/Data_preprocessing/test_img/'
    mask_path = '/content/drive/MyDrive/MaskGAN_demo/Data_preprocessing/test_label/'

    name_list = []

    for i in os.listdir(img_path):
        if i.split('.')[1] == 'jpg':
            name_list.append(i.split('.')[0])

    for name in name_list:
        img = cv2.imread(img_path+name+'.jpg')
        mask_m = cv2.imread(mask_path+name+'.png')
        mask = mask_m.copy()

        params = get_params(opt, (512,512))
        transform_mask = get_transform(opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
        transform_image = get_transform(opt, params)

        mask = transform_mask(Image.fromarray(np.uint8(mask))) 
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(Image.fromarray(np.uint8(img)))

        generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))
        
        save_image((generated.data[0] + 1) / 2,'./results/'+name+'.jpg')



        # np_arr = np.array((generated.data[0] + 1) / 2, dtype=np.uint8)
        # img = Image.fromarray(np_arr)
        # img.save('path')

    
