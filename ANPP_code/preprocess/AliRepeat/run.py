import numpy as np
import pandas as pd
import os
import sys

sys.path.append("../..")

from Config import *
from utils.Time import *


arr = sys.argv
folder_dataset = arr[1]
file_user_order_norm = "user_order_norm.csv"
flag_convert_to_norm = int(arr[2])
flag_remap_id = int(arr[3])

file_user_order = "user_order.csv"
file_meta = "meta.csv"
file_join = "user_order_join.csv"

def main():
    if(flag_convert_to_norm):
        cmd = "python -u convert_to_norm.py %s %s %s" % (folder_dataset, Config.pkl_review, Config.pkl_meta)
        os.system(cmd)


    if(flag_remap_id):
        cmd = "python -u remap_id.py %s %s %s %s %s %s" % (folder_dataset, Config.pkl_review, Config.pkl_meta, folder_dataset, Config.file_join, Config.pkl_remap)
        print(cmd)
        os.system(cmd)

main()