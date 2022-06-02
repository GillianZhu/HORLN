import os
import time
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("..")
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.electricity_dataset import ElectricityDataset
from util.electricity_evaluation import evaluation


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    opt_train = TrainOptions().parse()  # get training options

    dataset_train = ElectricityDataset(opt_train.dataroot, opt_train.phase, shuffle=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(dataset_train),
        shuffle=not opt_train.serial_batches,
        num_workers=int(opt_train.num_threads))
    dataset_train_size = len(dataset_train)
    print('The number of training samples = %d' % dataset_train_size)

    dataset_test = ElectricityDataset(opt.dataroot, opt.phase, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=len(dataset_test),
        shuffle=False,
        num_workers=int(opt.num_threads))
    dataset_test_size = len(dataset_test)
    print('The number of training samples = %d' % dataset_test_size)

    name = "RF"
    web_dir = os.path.join(opt.results_dir, name)  # define the website directory
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)

    trad_start_time = time.time()
    dataset_train_iter = iter(dataloader_train)
    data_train = dataset_train_iter.__next__()
    dataset_test_iter = iter(dataloader_test)
    data_test = dataset_test_iter.__next__()
    elec_data_test = data_test['elec']
    flag_data_test = data_test['flag']


    elec_data = data_train['elec']
    flag_data = data_train['flag']
    path_data = data_train['path']

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("ready to start")

    data_size_train = elec_data.shape[0]
    data_size_test = elec_data_test.shape[0]

    train_start_time = time.time()

    params = {'max_depth': 19, 'min_child_weight': 10, 'booster': 'gbtree'}
    XGBModel = xgb.XGBClassifier()
    XGBModel.set_params(**params)
    XGBModel = XGBModel.fit(elec_data.cpu().numpy().reshape(data_size_train, -1), flag_data.reshape(-1, 1).squeeze(1).cpu().numpy())

    print("train time:%.2f" % (time.time() - train_start_time))

    x_pred = XGBModel.predict(elec_data_test.cpu().numpy().reshape(data_size_test, -1))

    test_start_time = time.time()

    x_pred_probas = XGBModel.predict_proba(elec_data_test.cpu().numpy().reshape(data_size_test, -1))

    print("test time:%.2f" % (time.time() - test_start_time))

    all_flag = flag_data_test.squeeze(-1).cpu().numpy()

    # print("all pred, flag shape:", all_pred.shape, all_flag.shape)  # (16945,) (16945,)
    dataframe = pd.DataFrame({'pred': x_pred_probas[:, 1], 'flag': all_flag})
    dataframe = dataframe.sort_values(by=['pred'], ascending=False)
    print(dataframe)
    dataframe.to_csv(web_dir + "/result_probas.csv", sep=',')

    best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP = evaluation(dataframe)
    print("======pred probas=======")
    print("best_f1_threshold:", best_f1_threshold)
    print("best_f1_score:", best_f1_score)
    print("best_f1_precision:", best_f1_precision)
    print("best_f1_recall:", best_f1_recall)
    print("elec_auc:", elec_auc)
    print("MAP:", MAP)