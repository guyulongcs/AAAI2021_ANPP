import os

from Config import *

def main():
    if(Config.flag_preprocess_dataset):
        cmd = "cd preprocess; python -u preprocess.py %s; cd .." % (Config.dataset_run)
        print(cmd)
        os.system(cmd)

    if(Config.flag_run_model):
        cmd = "python -u run_dataset_model.py -dataset %s -model %s" % (Config.dataset_run, Config.model_run)
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    main()
