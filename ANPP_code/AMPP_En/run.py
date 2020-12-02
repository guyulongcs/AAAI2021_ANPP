import os, time, argparse
import sys
sys.path.append("..")
from Config import *


def main():
    args = sys.argv
    assert (len(args) == 1 + 5)
    dataset, model, folder_dataset, folder_dataset_model, test_neg_N_item = args[1], args[2], args[3], args[4], int(args[5])

    #rebuild model dataset
    #if(Config.flag_rebuild_model_dataset):
    #    cmd = "python -u build_dataset.py %s %s %d" % (folder_dataset, folder_dataset_model, test_neg_N_item)
    #    print(cmd)
    #    os.system(cmd)

    #train
    if(Config.flag_run_model_dataset):
        train_type = Config.train_type
        weight_loss = Config.weight_loss

        file_log_train = os.path.join(folder_dataset_model, "log_train_%s_%s_%s" % (train_type, weight_loss, Config.weight_loss_regularization))
        cmd = "python -u train.py --dataset %s --model %s --folder_dataset %s --folder_dataset_model %s --test_neg_N_item %d --train_type %s --weight_loss %s > %s" % (dataset, model, folder_dataset, folder_dataset_model, test_neg_N_item, train_type, weight_loss, file_log_train)
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    main()
