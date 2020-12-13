import os
import numpy as np

class Config():
    ##data
    #folder= "~/data/ANPP_dataset"
    folder="/Users/guyulong/program/project/AAAI2021_ANPP/ANPP_dataset"
    dataset_list_Amazon = ["Beauty", "Clothing_Shoes_and_Jewelry"]

    ##exp conf
    dataset_list = ["Beauty", "Clothing_Shoes_and_Jewelry"]
    data_list_real = ["financial", "mimic"]
    data_list_synthetic = ["hawkes"]

    flag_train_mini = False
    flag_test_mini = False
    flag_test_mini_cnt = 500

    item_type_list = ["item", "category", "all"]
    train_type = item_type_list[1]
    flag_train_cate = True
    flag_padding = True
    #dataset
    ##Hawkes
    dataset_run = "hawkes"
    ##Financial
    #dataset_run = "financial"
    ##MIMIC
    #dataset_run = "mimic"
    ##Beauty
    #dataset_run = "Beauty"
    ##Clothes
    #dataset_run = "Clothing_Shoes_and_Jewelry"

    #model
    model_run = "ANPP"

    ###preprocess
    flag_preprocess_dataset = False
    flag_download_data_Amazon = True
    flag_convert_pd = True
    flag_remap_id = True
    flag_build_dataset_user = True
    flag_build_dataset_train = True

    flag_run_model = True

    ###run dataset_model
    flag_rebuild_model_dataset = False
    flag_run_model_dataset = True

    dataset_should_process_time_seq_to_startZero = ["so"]
    dataset_should_process_time_seq_to_startZero.extend(dataset_list)

    time_method_list = ["time_norm_by_TimeMinMax", "time_org", "time_parse_start_zero", ]
    time_method = "time_norm_by_TimeMinMax"
    #time_method = "time_parse_start_zero"
    #time_method = "time_org"

    #seq
    if(dataset_run in dataset_should_process_time_seq_to_startZero):
       time_method = "time_parse_start_zero"
    max_seq_len = 50

    #bptt
    bptt_default=20
    bptt_dict = {
        "mimic": 2,
        "financial": 3,
        "hawkes": 2
    }
    bptt = bptt_dict.get(dataset_run, bptt_default)


    batch_size_default=64
    batch_size_dict = {
        "hawkes": 8,
        "financial":8,
    }
    batch_size = batch_size_dict.get(dataset_run, batch_size_default)

    metrics_K_default=20
    metrics_K_dict = {
        "mimic":20,
        "financial": 1,
        "hawkes": 1,
    }
    metrics_K = metrics_K_dict.get(dataset_run, metrics_K_default)

    trackperiod = 100

    #train_num_epochs
    train_num_epochs_default = 10
    train_num_epochs_dict = {
        "mimic": 100,
        "financial": 10,
        "hawkes": 20,
        "Beauty": 10,
        "Clothing_Shoes_and_Jewelry": 20
    }
    train_num_epochs = train_num_epochs_dict.get(dataset_run, train_num_epochs_default)
    #train_num_epochs=2

    ##split data
    flag_split_user_seq_OnlyOneTrainFromLast = False
    split_sequence_time_gap_max = 60
    split_sequence_time_T = 30
    sequence_mini_min_len=3

    #train
    #loss: event, time
    weight_loss = "1,1"
    weight_loss_regularization=0.001

    time_bucket_dim = 20

    time_scale_default = 0.01
    time_scale_dict = {
        "financial": 0.001,
        "hawkes": 0.001
    }
    time_scale = time_scale_dict.get(dataset_run, time_scale_default)

    learning_rate = 0.001

    #evaluation
    test_neg_N_item=100
    test_neg_N_cate=100

    eval_every_num_epochs = 1
    early_stopping_epochs = 10

    # pkl
    pkl_review = "reviews.pkl"
    pkl_meta = "meta.pkl"
    pkl_remap = "remap.pkl"

    file_join = "reviews_meta_join.csv"
    # pkl_train = "train.pkl"
    # pkl_train_split = "train_split.pkl"


    #dict_pkl_dataet
    dict_pkl_dataset = {
        "dataset": "dataset.pkl",
        "dataset_train": "dataset_train.pkl",
        "dataset_splitGap": "dataset_splitGap.pkl",
        "dataset_seq": "dataset_seq.pkl",
        "ANPP": "dataset_%s.pkl" % "ANPP",
    }

    file_dataset_desc = "desc"

    #output
    flag_debug = False

    #time encoding method
    time_encode_method_desc = {
        "pos": "encode by pos in seq",
        "time_direct": "map time to embedding directly",
        "time_delta_prev_direct": "map delta prev to embedding directly",
        "time_delta_prev_bucket": "map delta prev to bucket, then to embedding"
    }
    time_encode_method = "time_delta_prev_bucket"
    #time_encode_method = "time_delta_prev_direct"
    #time_encode_method = "pos"
    time_parse_log = False

    seed_num = 100

    num_blocks = 1
    num_heads = 4

    # IntensityRNN
    loss_time_method_list=["intensity", "mse", "gaussian"]
    loss_time_method = "mse"
    loss_time_method = "intensity"
    #loss_time_method = "gaussian"
    #event, time
    weight_loss_intensityRNN = "1,0.1"
    sigma_square = 10
    flagMinMax=True
    flagScale=True
    time_flag_parse_seq_to_startZero = True

    model_note=model_run

    hidden_units = 64


