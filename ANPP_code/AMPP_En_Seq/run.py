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
        file_log = os.path.join(folder_dataset, "log_dataset")
        cmd = "python -u build_dataset.py %s %s > %s" % (folder_dataset, folder_dataset_model, file_log)
        print(cmd)
        os.system(cmd)

    #train
    if(Config.flag_run_model_dataset):
        #file_log_train = os.path.join(folder_dataset_model, "log_train")

        file_log_train = os.path.join(folder_dataset_model, "log_train_WeightLoss[%s]_TP[%s]_TimeEncode[%s]_LossTime[%s]_T[%s]_K[%s]_Ep[%s]_Bl[%s]_Hea[%s]_Hid[%s]_Model[%s]" % (Config.weight_loss, Config.time_method, Config.time_encode_method, Config.loss_time_method, str(Config.bptt), str(Config.metrics_K), str(Config.train_num_epochs), str(Config.num_blocks), str(Config.num_heads),  str(Config.hidden_units), Config.model_note))
        cmd = "python -u run_model.py --dataset %s --model %s --folder_dataset %s --folder_dataset_model %s --max_T %d --weight_loss %s > %s" % (dataset, model, folder_dataset, folder_dataset_model, Config.bptt, Config.weight_loss, file_log_train)
        print(cmd)
        os.system(cmd)



if __name__ == "__main__":
    main()
