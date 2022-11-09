import sys
import cv2
import time
from PIL import Image
from torchvision.utils import save_image
from data.base_dataset import BaseDataset, get_params, get_transform, normalize


import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
from torchvision import transforms, utils

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import models.networks as networks
import util.util as util
# from util.visualizer import Visualizer

opt = TrainOptions().parse()

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
device = 'cuda'
model = model.to(device)

# visualizer = Visualizer(opt)
# if opt.distributed:
#         torch.cuda.set_device(args.local_rank)
#         torch.distributed.init_process_group(backend="nccl", init_method="env://")
#         synchronize()

# optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
netVAE = networks.define_VAE(opt.input_nc)
networks.print_network(netVAE)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        
        # print("DATA.shape:  ", data)
        inter_label_1, label = data['inter_label_1'], data['label']
        inter_label_2, label_ref = data['inter_label_2'], data['label_ref']
        image, image_ref, path, path_ref = data['image'], data['image_ref'], data['path'], data['path_ref']        # print(image.shape, label.shape, image_ref.shape, label_ref.shape)

        ############## Forward Pass ######################
        # data['inst'], data['feat'],
        device = 'cuda'
        model = model.to(device)
        losses, generated = model(inter_label_1, label, inter_label_2, image, label_ref, image_ref, infer=save_fake)
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        # if opt.fp16:                                
        #     with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        # else:
        loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        # if opt.fp16:                                
        #     with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        # else:
        loss_D.backward()        
        optimizer_D.step()        

        ############## Display results and errors ##########
#         if i % 100 == 0:
#             with torch.no_grad():
#                 g_ema.eval()
#                 sample, _ = g_ema([sample_z])
#                 utils.save_image(
#                     sample,
#                     f"sample512/{str(i).zfill(6)}.png",
#                     nrow=int(args.n_sample ** 0.5),
#                     normalize=True,
#                     range=(-1, 1),
#                 )

#         if i % 10000 == 0:
#             torch.save(
#                 {
#                      "g": g_module.state_dict(),
#                      "d": d_module.state_dict(),
#                      "g_ema": g_ema.state_dict(),
#                      "g_optim": g_optim.state_dict(),
#                      "d_optim": d_optim.state_dict(),
#                      "args": args,
#                      "ada_aug_p": ada_aug_p,
#                  },
#                 f"checkpoint512/{str(i).zfill(6)}.pt",
#             )
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
#             visualizer.print_current_errors(epoch, epoch_iter, errors, t)
#             visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        #if save_fake:
            #visuals = OrderedDict([('input_label', util.tensor2label(data['label_ref'][0], opt.label_nc)),
                                   #('synthesized_image', util.tensor2im(generated.data[0])),
                                   #('real_image', util.tensor2im(data['image_ref'][0]))])
#             visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate() 