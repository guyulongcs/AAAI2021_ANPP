import os, time, argparse
from Config import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='Beauty', help='dataset')
    parser.add_argument('-model',  default='atrank', help='model')


    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    print "dataset:", dataset
    print "model:", model

    folder_dataset = os.path.join(Config.folder, dataset)
    folder_dataset_model = os.path.join(folder_dataset, model)

    cmd = "mkdir -p %s; cd %s; python -u run.py %s %s %s %s %d; cd .." % (folder_dataset_model, model,
                                                dataset, model, folder_dataset, folder_dataset_model, Config.test_neg_N_item)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()