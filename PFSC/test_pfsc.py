import sys
import os
import time
sys.path.append("..")
from options.test_options import TestOptions
from data.electricity_dataset import ElectricityDataset
from models import create_model
from util.electricity_evaluation import evaluation
import pandas as pd
import random
import torch.utils.data as data
import torch
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataroot = opt.dataroot
    dataset_test = ElectricityDataset(dataroot, 'test', shuffle=False)
    dataloader_for_meta_cls = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_threads))
    dataset_size_for_meta_cls = len(dataloader_for_meta_cls)  # get the number of images in the dataset.
    print('The number of testing samples for meta cls = %d' % dataset_size_for_meta_cls)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}/'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)

    test_start_time = time.time()
    model.eval()
    all_pred = []
    all_flag = []
    all_path = []
    print('opt.batch_size: ', opt.batch_size)
    test_sample_num = 0
    for i, data in enumerate(dataloader_for_meta_cls):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        test_sample_num = test_sample_num + len(visuals['prediction'])
        for j in range(len(visuals['prediction'])):
           pred = visuals['prediction'][j].cpu().float().numpy()[0] 
           flag = visuals['flag'][j].cpu().float().numpy()[0] 
           path = visuals['image_paths'][j] 

           if 'week_pred' in visuals.keys():
              week_pred = visuals['week_pred'][j].cpu().float().numpy()[0] 
              day_pred  = visuals['day_pred'][j].cpu().float().numpy()[0] 
              pred = pred #0.9 * pred + 0.0 * week_pred + 0.1 * day_pred

           all_pred.append(pred)
           all_flag.append(flag)        
           all_path.append(path)

        if i % 10 == 0: 
            print('processing (%04d)-th sample... %s' % (test_sample_num, path))


    dataframe = pd.DataFrame({'pred': all_pred, 'flag':all_flag, 'path':all_path})
    dataframe = dataframe.sort_values(by=['pred'], ascending=False)
    dataframe.to_csv(web_dir+"result.csv",index=False,sep=',')

    # best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP_all, MAP100, MAP200, MAP300, MAP500 = evaluation(dataframe)
    best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP_all = evaluation(dataframe)
    print("best_f1_threshold:", best_f1_threshold)
    print("best_f1_precision:", best_f1_precision)
    print("best_f1_recall:", best_f1_recall)
    print("\n")
    
    print("best_f1_score:", best_f1_score)
    print("elec_auc:", elec_auc)
    print("MAP_all:", MAP_all)

    print('End of testing \t Time Taken: %d sec' % (time.time() - test_start_time))

    with open(os.path.join(opt.results_dir, opt.name) + '/'+ opt.name + "_evaluation.csv", "a") as f:
        evaluation_str = 'Epoch, ' + str(opt.epoch) + \
                         ', Threshold, ' + str(best_f1_threshold) + \
                         ', Recall, ' + str(best_f1_recall) + \
                         ', Precision, ' + str(best_f1_precision) + \
                         ', F1_Score, ' + str(best_f1_score) + \
                         ', AUC, ' + str(elec_auc) + \
                         ', MAP, ' + str(MAP_all) + '\n'

        f.writelines(evaluation_str) 
