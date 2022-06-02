import os
from options.test_options import TestOptions
from util.electricity_evaluation import evaluation
import pandas as pd


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}/'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('web directory', web_dir)
    if not os.path.exists(web_dir):
        print('no directory:', web_dir)
    else:
        print('opt.batch_size: ', opt.batch_size)
        pred_path = web_dir + "result.csv"
        dataframe = pd.read_csv(pred_path)

        best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP = evaluation(dataframe, opt.best_threshold, trad=True)
        print("best_f1_threshold:", best_f1_threshold)
        print("best_f1_precision:", best_f1_precision)
        print("best_f1_recall:", best_f1_recall)
        print("best_f1_score:", best_f1_score)
        print("elec_auc:", elec_auc)
        print("MAP:", MAP)

        with open(os.path.join(opt.results_dir, opt.name) + '/'+ opt.name + "_evaluation_thres.csv", "a") as f:
            evaluation_str = 'Epoch, '       + str(opt.epoch) + \
                         ', Threshold, ' + str(best_f1_threshold) + \
                         ', Recall, '    + str(best_f1_recall) + \
                         ', Precision, ' + str(best_f1_precision) + \
                         ', F1_Score, '  + str(best_f1_score) + \
                         ', AUC, '       + str(elec_auc) + \
                         ', MAP, '       + str(MAP) + '\n'

            f.writelines(evaluation_str)
