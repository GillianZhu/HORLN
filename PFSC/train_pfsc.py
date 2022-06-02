import sys
import time
import numpy as np
import os
import pickle
import random
import torch
import torch.utils.data as data
sys.path.append("..")
from data.electricity_dataset import ElectricityDataset, DatasetForMetaCls
from options.train_options import TrainOptions
from models import create_model

'''
####### PFSC #######
based on stacked_generalization; contains three xgb classifiers as base classifiers and a TCN as the meta classifier
the predictions from base classifiers will be the inputs of the meta classifier, whose outputs will be the final results

step1: load normal and abnormal training samples and divide into five folds respectively; load validation and testing sets
step2: train xgb classifiers, and use the trained model to generate inputs for meta classifiers on val and test datasets
step3: combine predictions of base classifiers on the training dataset; combine predictions of base classifiers on
the val and test datasets and save the results to files for asynchronous val and test
step4: train TCN on training set and save the trained models of each epoch
step5: asynchronous val and test: input the saved predictions of base classifiers on the val and test datasets to the 
trained models to get final val and test results
'''


def get_chunk_data(dict_train_normal, dict_train_abnormal, fold_num=5):
    start_time = time.time()
    Xtrain_normal = dict_train_normal['elec']
    Ytrain_normal = dict_train_normal['flag']
    Xtrain_abnormal = dict_train_abnormal['elec']
    Ytrain_abnormal = dict_train_abnormal['flag']
    train_paths, train_y = None, None
    chunk_data_dict_list, x_val_list = [], []
    Xtrain_normal = Xtrain_normal.reshape(Xtrain_normal.shape[0], -1).chunk(chunks=fold_num, dim=0)
    Ytrain_normal = Ytrain_normal.reshape(-1, 1).squeeze(1).chunk(chunks=fold_num, dim=0)
    paths_train_normal = [dict_train_normal['path'][i * len(Xtrain_normal[0]):
                                                    (i + 1) * len(Xtrain_normal[0])] for i in range(fold_num - 1)] + \
                         [dict_train_normal['path'][(fold_num - 1) * len(Xtrain_normal[0]):]]
    Xtrain_abnormal = Xtrain_abnormal.reshape(Xtrain_abnormal.shape[0], -1).chunk(chunks=fold_num, dim=0)
    Ytrain_abnormal = Ytrain_abnormal.reshape(-1, 1).squeeze(1).chunk(chunks=fold_num, dim=0)
    paths_train_abnormal = [dict_train_abnormal['path'][i * len(Xtrain_abnormal[0]):
                                                        (i + 1) * len(Xtrain_abnormal[0])] for i in
                            range(fold_num - 1)] + \
                           [dict_train_abnormal['path'][(fold_num - 1) * len(Xtrain_abnormal[0]):]]
    for i in range(fold_num):
        x_val = torch.cat([Xtrain_normal[i], Xtrain_abnormal[i]])
        y_val = torch.cat([Ytrain_normal[i], Ytrain_abnormal[i]])
        path_val = paths_train_normal[i] + paths_train_abnormal[i]
        x_train = torch.cat([torch.cat([Xtrain_normal[j] for j in range(fold_num) if j != i]),
                             torch.cat([Xtrain_abnormal[j] for j in range(fold_num) if j != i])])
        y_train = torch.cat([torch.cat([Ytrain_normal[j] for j in range(fold_num) if j != i]),
                             torch.cat([Ytrain_abnormal[j] for j in range(fold_num) if j != i])])
        path_train = []
        for j in range(fold_num):
            if j != i:
                path_train += paths_train_normal[j]
        for j in range(fold_num):
            if j != i:
                path_train += paths_train_abnormal[j]
        data_dict = {'elec': x_train, 'flag': y_train, 'path': path_train}
        chunk_data_dict_list.append(data_dict)
        x_val_list.append(x_val)
        if train_y is None:
            train_y = y_val
            train_paths = path_val
        else:
            train_y = torch.cat([train_y, y_val], dim=0)
            train_paths += path_val
    print('chunk \t Time Taken: %.2f sec' % (time.time() - start_time))
    return chunk_data_dict_list, x_val_list, train_y, train_paths


def XGB(chunk_data_dicts, x_vals, Xval, Xtest, fold_num=5, params=None):
    import xgboost as xgb
    Xval = Xval.reshape(Xval.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)
    pred_train, pred_val, pred_test = None, \
                                      np.zeros((Xval.shape[0], 2)), \
                                      np.zeros((Xtest.shape[0], 2))
    models = []
    params = {'max_depth': 19, 'min_child_weight': 10, 'booster': 'gbtree'} if params is None else params

    start_time = time.time()

    for i in range(fold_num):
        one_start_time = time.time()
        x_val = x_vals[i]
        data_dict = chunk_data_dicts[i]
        train_data = DatasetForMetaCls(data_dict, 'train')
        dataloader_train = torch.utils.data.DataLoader(
            train_data,
            batch_size=len(train_data),
            shuffle=True,
            num_workers=4)
        dataset_train_iter = iter(dataloader_train)
        data_train = dataset_train_iter.__next__()
        elec_data = data_train['elec']
        flag_data = data_train['flag']

        XGBModel = xgb.XGBClassifier()
        XGBModel.set_params(**params)
        XGBModel.fit(elec_data.cpu().numpy(), flag_data.reshape(-1, 1).squeeze(1).cpu().numpy())
        models.append(XGBModel)
        x_pred_proba = XGBModel.predict_proba(x_val.cpu().numpy())
        x_val_pred_proba = XGBModel.predict_proba(Xval.cpu().numpy())
        x_test_pred_proba = XGBModel.predict_proba(Xtest.cpu().numpy())

        if pred_train is None:
            pred_train = x_pred_proba
        else:
            pred_train = np.concatenate([pred_train, x_pred_proba], axis=0)
        pred_test += x_test_pred_proba * 1.0 / fold_num
        pred_val += x_val_pred_proba * 1.0 / fold_num
        print('one XGB \t Time Taken: %.2f sec' % (time.time() - one_start_time))

    print('End of XGB \t Time Taken: %.2f sec' % (time.time() - start_time))

    return models, pred_train, pred_val, pred_test


def save_base_preds(elec_data, flag_data, path_data, run_type='test', save_path = 'datasets/electricity/pfsc'):
    f_normal = open(os.path.join(save_path, 'normal_'+run_type+'.txt'), "a")
    f_abnormal = open(os.path.join(save_path, 'abnormal_'+ run_type+'.txt'), "a")
    for p in range(len(path_data)):
        np.savez(os.path.join(save_path, path_data[p]), elec=elec_data[p], flag=flag_data[p], path=path_data[p])
        if flag_data[p] == 1:
            f_abnormal.write(path_data[p]+"\n")
        else:
            f_normal.write(path_data[p]+"\n")
    f_normal.close()
    f_abnormal.close()
    return


def save_trad_models(name, models, save_dir):
    for i, model in enumerate(models):
        save_path = os.path.join(save_dir, name + "_" + str(i) + ".model")
        s = pickle.dumps(model)
        f = open(save_path, "wb+")
        f.write(s)
        f.close()
    return


def combine_base_preds(names, preds=None, save_model=True, models=None, save_dir=None):
    elec = None
    for i in range(len(names)):
        pred_tensor = torch.from_numpy(preds[i]).type(torch.FloatTensor)
        if save_model:
            save_trad_models(name=names[i], models=models[i], save_dir=save_dir)
        if elec is None:
            elec = pred_tensor
        else:
            elec = torch.cat([elec, pred_tensor], dim=1)
    return elec.unsqueeze(1)


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    ### load normal and abnormal training samples and divide into five folds respectively
    dataroot = opt.dataroot
    data_train_normal = ElectricityDataset(dataroot, 'train', shuffle=False, type='normal')
    data_train_abnormal = ElectricityDataset(dataroot, 'train', shuffle=False, type='abnormal')
    data_normal_size = 15502
    data_abnormal_size = 1446
    dataset_train_normal = torch.utils.data.DataLoader(data_train_normal, batch_size=data_normal_size, shuffle=False, num_workers=4)
    dataset_train_normal_iter = iter(dataset_train_normal)
    data_train_n = dataset_train_normal_iter.__next__()
    dataset_train_abnormal = torch.utils.data.DataLoader(data_train_abnormal, batch_size=data_abnormal_size, shuffle=False, num_workers=4)
    dataset_train_abnormal_iter = iter(dataset_train_abnormal)
    data_train_a = dataset_train_abnormal_iter.__next__()
    chunk_data, xs, ys, paths = get_chunk_data(data_train_n, data_train_a)  # divide train data into five folds

    ### load validation and testing sets
    dataset2 = ElectricityDataset(dataroot, 'test', shuffle=False)
    dataset_test = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2), shuffle=False, num_workers=4)
    dataset_test_iter = iter(dataset_test)
    data_test = dataset_test_iter.__next__()
    elec_data_test = data_test['elec']
    flag_data_test = data_test['flag']
    path_data_test = data_test['path']

    dataset3 = ElectricityDataset(dataroot, 'val', shuffle=False)
    dataset_val = torch.utils.data.DataLoader(dataset3, batch_size=len(dataset3), shuffle=False, num_workers=4)
    dataset_val_iter = iter(dataset_val)
    data_val = dataset_val_iter.__next__()
    elec_data_val = data_val['elec']
    flag_data_val = data_val['flag']
    path_data_val = data_val['path']

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_save_dir = os.path.join('../datasets', opt.name)
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)
        os.mkdir(os.path.join(data_save_dir, "val"))
        os.mkdir(os.path.join(data_save_dir, "test"))
        os.mkdir(os.path.join(data_save_dir, "val", "normal"))
        os.mkdir(os.path.join(data_save_dir, "val", "abnormal"))
        os.mkdir(os.path.join(data_save_dir, "test", "normal"))
        os.mkdir(os.path.join(data_save_dir, "test", "abnormal"))

    train_start_time = time.time()
    ### train xgb classifiers, and use the trained model to generate inputs for meta classifiers on val and test datasets
    xgb_models, xgb_pred_proba, xgb_pred_proba_val, xgb_pred_proba_test = XGB(chunk_data,
                                                                              xs,
                                                                                 elec_data_val,
                                                                                 elec_data_test)
    params = {'max_depth': 16, 'learning_rate': 0.3, 'scale_pos_weight': 1, 'booster': 'gbtree'}
    xgb_models1, xgb_pred_proba1, xgb_pred_proba_val1, xgb_pred_proba_test1 = XGB(chunk_data,
                                                                              xs,
                                                                              elec_data_val,
                                                                              elec_data_test,
                                                                              params=params)

    params = {'max_depth': 16, 'learning_rate': 0.2, 'scale_pos_weight': 1, 'booster': 'gbtree'}
    xgb_models2, xgb_pred_proba2, xgb_pred_proba_val2, xgb_pred_proba_test2 = XGB(chunk_data,
                                                                              xs,
                                                                                  elec_data_val,
                                                                                  elec_data_test,
                                                                                  params=params)
    ### combine predictions of base classifiers on the training dataset
    elec_train = combine_base_preds(names=["xgb_1", "xgb_2", "xgb_3"],
                                    preds=[xgb_pred_proba, xgb_pred_proba1, xgb_pred_proba2],
                                    save_model=True,
                                    models=[xgb_models, xgb_models1, xgb_models2],
                                    save_dir=save_dir)
    print("elec:", elec_train.shape)

    data_dict = {"elec": elec_train.type(torch.FloatTensor), "flag": ys.reshape(-1, 1).type(torch.FloatTensor), "path": paths}
    dataset_for_meta_cls = DatasetForMetaCls(data_dict, opt.phase)
    dataloader_for_meta_cls = torch.utils.data.DataLoader(
        dataset_for_meta_cls,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))
    dataset_size_for_meta_cls = len(dataset_for_meta_cls)  # get the number of images in the dataset.
    print('The number of training samples for meta cls = %d' % dataset_size_for_meta_cls)

    ### combine predictions of base classifiers on the val and test dataset and save the results to files for asynchronously val and test
    elec_val = combine_base_preds(names=["xgb_1", "xgb_2", "xgb_3"],
                                  preds=[xgb_pred_proba_val, xgb_pred_proba_val1, xgb_pred_proba_val2],
                                  save_model=False)
    print('The number of validation samples = {}, shape: {}'.format(len(elec_val), elec_val.shape))
    save_base_preds(elec_val, flag_data_val, path_data_val, run_type='val', save_path=os.path.join(data_save_dir, 'val'))

    elec_test = combine_base_preds(names=["xgb_1", "xgb_2", "xgb_3"],
                                   preds=[xgb_pred_proba_test, xgb_pred_proba_test1, xgb_pred_proba_test2],
                                   save_model=False)
    print('The number of test samples = {}, shape: {}'.format(len(elec_test), elec_test.shape))
    save_base_preds(elec_test, flag_data_test, path_data_test, run_type='test', save_path=os.path.join(data_save_dir, 'test'))

    ### train TCN on training set
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        for i, data in enumerate(dataloader_for_meta_cls):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.10f ' % (k, v)
                print(message)  # print the message

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    print('End of training \t Time Taken: %d sec' % (time.time() - train_start_time))
