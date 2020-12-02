import os
import sys

#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0,parentdir)

sys.path.append("..")
from Config import *


arr = sys.argv
assert(len(arr) == 1+1)
dataset = arr[1]
folder_dataset = os.path.join(Config.folder, dataset)


if dataset in Config.dataset_list_Amazon:
    print "\npreprocess dataset(Amazon)..."
    folder_dataset = os.path.join(Config.folder, dataset)
    print "\npreprocess dataset:%s..." % dataset
    file_review = "reviews_%s_5.json" % dataset
    file_meta = "meta_%s.json" % dataset

    file_join = "reviews_meta_join.csv"


    cmd = "mkdir -p %s" % folder_dataset
    os.system(cmd)

    if(Config.flag_download_data_Amazon):
        cmd = "sh 0_download_raw.sh %s %s %s" % (folder_dataset, file_review, file_meta)
        os.system(cmd)

    if(Config.flag_convert_pd):
        cmd = "python -u 1_convert_pd.py %s %s %s %s %s %s" % (folder_dataset, file_review, file_meta, folder_dataset, Config.pkl_review, Config.pkl_meta)
        os.system(cmd)

    if (Config.flag_remap_id):
        cmd = "python -u 2_remap_id.py %s %s %s %s %s %s" % (
        folder_dataset, Config.pkl_review, Config.pkl_meta, folder_dataset, file_join, Config.pkl_remap)
        print(cmd)
        os.system(cmd)

if dataset == "AliRepeat" and (Config.flag_convert_pd or Config.flag_remap_id):
    print "\npreprocess dataset %s..." % dataset
    folder_dataset = os.path.join(Config.folder, dataset)

    cmd = "cd %s; python -u run.py %s %d %d; cd .." % (dataset, folder_dataset, int(Config.flag_convert_pd) , int(Config.flag_remap_id))
    print(cmd)
    os.system(cmd)




if(Config.flag_build_dataset_user):
    cmd = "python -u 3_build_dataset_user.py %s %s %s %s %d" % (
    folder_dataset, Config.pkl_remap, folder_dataset, Config.test_neg_N_item, Config.test_neg_N_cate)
    print(cmd)
    os.system(cmd)

if(Config.flag_build_dataset_train):
    cmd = "python -u 4_build_dataset_train.py %s %s %s %d" % (
    folder_dataset, folder_dataset, Config.test_neg_N_item, Config.test_neg_N_cate)
    print(cmd)
    os.system(cmd)


#exit(0)

if (dataset in Config.data_list_real) or (dataset in Config.data_list_synthetic) or (dataset in Config.dataset_list_Amazon):
    print "\npreprocess dataset %s..." % dataset
    folder_dataset = os.path.join(Config.folder, dataset)

    #cmd = "python -u build_dataset_restore_pkl.py %s" % (folder_dataset)

    cmd = "python -u build_dataset_seq.py %s" % (folder_dataset)

    print(cmd)
    os.system(cmd)

    print "\npreprocess dataset:%s done!" % dataset



