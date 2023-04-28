"""
# training LIR for UNIR
# The codes is implemented by "UNIT", double encoder branches and self-supervised contraints are added for training.
# Author: Wenchao. Du
# Time: 2019. 08
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, data_prefetcher
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil
import random
from skimage.measure import compare_psnr, compare_ssim
import torchvision.utils as vutils
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/LIR_config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", default=False)
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
parser.add_argument("--data_type", default='realworld')
opts = parser.parse_args()

def main():
    cudnn.benchmark = True
    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path
    config['data_type'] = opts.data_type

    if config['data_type'] == 'realworld':
        config['train_H'] = ['/home/lyp/3T/RESIDE/RTTS/UnannotatedHazyImages1']
        config['train_G'] = ['/home/lyp/3T/RESIDE/OTS_ALPHA/clear/clear_images8937/']

        config['val_H'] = '/home/lyp/3T/RESIDE/HSTS/synthetic/synthetic/'
        config['val_G'] = '/home/lyp/3T/RESIDE/HSTS/synthetic/original/'
        config['test_H']=['/home/lyp/3T/RESIDE/HSTS/synthetic/synthetic/']


    # Setup model and data loader
    trainer = UNIT_Trainer(config)
    if torch.cuda.is_available():
        trainer.cuda(config['gpuID'])
    train_loader_a, train_loader_b, val_loader_a, val_loader_b, test_loader_a= get_all_data_loaders(config)
    
    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    writer = SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    print('start training !!')
    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    
    TraindataA = data_prefetcher(train_loader_a) 
    TraindataB = data_prefetcher(train_loader_b)
    valdataA = data_prefetcher(val_loader_a)
    valdataB = data_prefetcher(val_loader_b)
    testdataA = data_prefetcher(test_loader_a)
    while True:
        dataA = TraindataA.next()
        dataB = TraindataB.next()
        if dataA is None or dataB is None:
            TraindataA = data_prefetcher(train_loader_a) 
            TraindataB = data_prefetcher(train_loader_b)
            dataA = TraindataA.next()
            dataB = TraindataB.next()

        # Main training code
        # print('start train')
        for _ in range(3):
            trainer.content_update(dataA, dataB, config)
        trainer.dis_update(dataA, dataB, config)
        trainer.gen_update(dataA, dataB, config)
        # torch.cuda.synchronize()
        trainer.update_learning_rate()
        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        # ---validation
        if (iterations + 1) % config['val_iter'] == 0:
            print('start validation !!')
            print('val_dataset_HSTS')
            trainer.eval()
            with torch.no_grad():
                psnr = 0
                ssim = 0
                val_ite = 0
                average_time = 0

                val_dataA = valdataA.next()
                val_dataB = valdataB.next()
                if val_dataA is None or val_dataB is None:
                    valdataA = data_prefetcher(val_loader_a)
                    valdataB = data_prefetcher(val_loader_b)
                    val_dataA = valdataA.next()
                    val_dataB = valdataB.next()
                # if 0:
                while val_dataA is not None and val_dataB is not None:
                    if config['data_type'] == 'realworld':
                        img_h = val_dataA.shape[2]
                        img_w = val_dataA.shape[3]
                        pad_h = img_h % 4
                        pad_w = img_w % 4
                        if pad_h != 0 or pad_w != 0:
                            val_dataA = val_dataA[:, :, 0:img_h - pad_h, 0:img_w - pad_w]
                            val_dataB = val_dataB[:, :, 0:img_h - pad_h, 0:img_w - pad_w]

                    start_time = time.time()
                    content = trainer.gen_a.encode_cont(val_dataA)
                    val_out = trainer.gen_b.dec_cont(content)  #
                    end_time = time.time() - start_time
                    average_time += end_time

                    val_O = val_out.clone()

                    val_results_path = './val_results'
                    if not os.path.exists(val_results_path):
                        os.mkdir(val_results_path)
                    vutils.save_image(val_O, val_results_path + '/%05d.png' % (int(val_ite+1)))
                    outputs = torch.squeeze(val_out)
                    outputs = outputs.permute(1, 2, 0).to('cpu', torch.float32).numpy()
                    val_dataB = torch.squeeze(val_dataB)
                    val_dataB = val_dataB.permute(1, 2, 0).to('cpu', torch.float32).numpy()

                    psnr += compare_psnr(val_dataB, outputs, data_range=None)
                    ssim += compare_ssim(val_dataB, outputs, multichannel=True)

                    val_ite += 1
                    val_dataA = valdataA.next()
                    val_dataB = valdataB.next()
                    print('{}:validation time is {}'.format(val_ite, end_time))
                psnr /= val_ite
                ssim /= val_ite
                average_time /= val_ite
                print('psnr:{}, ssim:{}'.format(psnr, ssim))
                print('average:validation time is {}'.format(average_time))
                writer.add_scalar('PSNR', psnr, iterations + 1)
                writer.add_scalar('SSIM', ssim, iterations + 1)
                # ----save psnr
                f = open("./psnr.txt", 'a')
                f.write('{}: psnr:{}, ssim:{}'.format(iterations + 1, psnr, ssim))
                f.write("\n")
                f.close()
                writer.add_scalar('PSNR', psnr, iterations + 1)
                writer.add_scalar('SSIM', ssim, iterations + 1)















                # -----test real
                test_real = testdataA.next()
                if test_real is None :
                    testdataA = data_prefetcher(test_loader_a)
                    test_real = testdataA.next()
                count_n = 0
                test_ite = 0
                while test_real is not None:
                    img_h = test_real.shape[2]
                    img_w = test_real.shape[3]
                    pad_h = img_h % 4
                    pad_w = img_w % 4
                    if pad_h != 0 or pad_w != 0:
                        test_real = test_real[:, :, 0:img_h - pad_h, 0:img_w - pad_w]

                    content = trainer.gen_a.encode_cont(test_real)
                    val_out = trainer.gen_b.dec_cont(content)  #

                    val_results = torch.cat([test_real, val_out], 0)

                    val_results_path = './test_results_real'
                    if not os.path.exists(val_results_path):
                        os.mkdir(val_results_path)
                    vutils.save_image(val_results, './test_results_real/{}_{}.png'.format(iterations + 1, count_n), normalize=False)
                    test_real = testdataA.next()
                    count_n += 1
                    print('test:', count_n)



            #val_2






            #val_3



            trainer.train()



        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(dataA, dataB)
            if image_outputs is not None:
                write_2images(image_outputs, display_size, image_directory, 'train_{}'.format(iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            writer.close()
            sys.exit('Finish training')
        

if __name__ == "__main__":
    main()

'''


'''
