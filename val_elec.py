"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.electricity_evaluation import evaluation
# from util import html
import pandas as pd
import torch.nn as nn


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    #opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}/'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)

    val_start_time = time.time()
    model.eval()
    all_pred = []
    all_flag = []
    all_path = []
    print('opt.batch_size: ', opt.batch_size)
    test_sample_num = 0
    loss_Sigmoid = nn.BCEWithLogitsLoss()
    val_loss = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        #visual_names = ['elec', 'meta', 'MIL_prediction', 'flag', 'image_paths']
        #visual_names = ['elec', 'meta', 'prediction', 'week_pred', 'day_pred', 'flag', 'image_paths']
        test_sample_num = test_sample_num + len(visuals['prediction'])
        for j in range(len(visuals['prediction'])):
            val_loss += loss_Sigmoid(visuals['prediction'][j], visuals['flag'][j]).item()
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

    val_loss = val_loss / ((i+1) * len(visuals['prediction']))
    print("val_loss:", val_loss)
    # best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP_all, MAP1percent, MAP10percent, MAP50percent, MAP80percent = evaluation(dataframe)
    best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP_all = evaluation(dataframe)
    print("best_f1_threshold:", best_f1_threshold)
    print("best_f1_precision:", best_f1_precision)
    print("best_f1_recall:", best_f1_recall)
    print("\n")
    
    print("best_f1_score:", best_f1_score)
    print("elec_auc:", elec_auc)
    # print("MAP1per:", MAP1percent)
    # print("MAP10per:", MAP10percent)
    # print("MAP50per:", MAP50percent)
    # print("MAP80per:", MAP80percent)
    print("MAP_all:", MAP_all)

    print('End of validating \t Time Taken: %d sec' % (time.time() - val_start_time))


    with open(os.path.join(opt.results_dir, opt.name) + '/'+ opt.name + "_validation.csv", "a") as f:
        evaluation_str = 'Epoch, ' + str(opt.epoch) + \
                         ', Threshold, ' + str(best_f1_threshold) + \
                         ', Recall, ' + str(best_f1_recall) + \
                         ', Precision, ' + str(best_f1_precision) + \
                         ', F1_Score, ' + str(best_f1_score) + \
                         ', AUC, ' + str(elec_auc) + \
                         ', MAP, ' + str(MAP_all) + '\n'

        f.writelines(evaluation_str)
