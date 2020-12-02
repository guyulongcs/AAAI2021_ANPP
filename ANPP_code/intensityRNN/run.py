import os, time, argparse
import sys
sys.path.append("..")
from Config import *


def main():
    args = sys.argv
    assert (len(args) == 1 + 5)
    dataset, model, folder_dataset, folder_dataset_model, test_neg_N_item = args[1], args[2], args[3], args[4], int(args[5])

    #rebuild model dataset
    if(Config.flag_rebuild_model_dataset):
        cmd = "python -u build_dataset.py %s %s" % (folder_dataset, folder_dataset_model)
        print(cmd)
        os.system(cmd)

    #train
    if(Config.flag_run_model_dataset):

        file_log_train = os.path.join(folder_dataset_model,
                                      "log_train_WLInt[%s]_TE[%s]_LossT[%s]_T[%s]_K[%s]_Ep[%s]_Model[%s]" % (
                                      Config.weight_loss_intensityRNN, Config.time_encode_method, Config.loss_time_method,
                                      str(Config.bptt), str(Config.metrics_K), str(Config.train_num_epochs),
                                      Config.model_note))

        cmd = "python -u run_model.py --dataset %s --model %s --folder_dataset %s --folder_dataset_model %s --bptt %d --scale %f > %s" % (dataset, model, folder_dataset, folder_dataset_model, Config.bptt, Config.time_scale, file_log_train)
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()
