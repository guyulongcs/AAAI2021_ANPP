import os
import numpy as np

class Config():
    ##data
    folder= "~/data/AMPP_PP"
    dataset_list_Amazon = ["Beauty", "Electronics", "Clothing_Shoes_and_Jewelry"]

    ##exp conf
    dataset_list = ["Beauty", "Electronics", "Clothing_Shoes_and_Jewelry", 'AliRepeat']
    data_list_real = ["ali", "book_order", "lastfm", "sns", "so", "mimic2", "twitter"]
    data_list_synthetic = ["hawkes", "hawkes5", "selfcorrecting", "mixture-HMM", "loglogistic"]

    model_list = ['rmtpp', 'rmtpp_att', 'intensityRNN', 'atrank', "SASRec", 'AMPP_En', 'AMPP_En_SASRec', 'AMPP_EnDe_ATRank', 'AMPP_EnDe', "AMPP_En_Seq", 'RNN', 'Transformer', "neuraHawkes"]

    flag_train_mini = False
    flag_test_mini = False
    flag_test_mini_cnt = 500

    item_type_list = ["item", "category", "all"]
    train_type = item_type_list[0]
    flag_train_cate = True
    flag_padding = True
    #dataset, model
    dataset_run = "so"
    dataset_run = "ali"
    dataset_run = "lastfm"
    dataset_run = "book_order"
    #dataset_run = "mimic2"
    #dataset_run = "hawkes"
    #dataset_run = "hawkes5"
    #dataset_run = "selfcorrecting"
    #dataset_run = "mixture-HMM"

    dataset_run = "Beauty"
    #dataset_run = "Electronics"
    #dataset_run = "Clothing_Shoes_and_Jewelry"
    #dataset_run = "AliRepeat"
    model_run = "AMPP_En_Seq"
    #model_run = "rmtpp"
    #model_run = "intensityRNN"
    #model_run = "RNN"
    #model_run = "Transformer"
    #model_run = "neuraHawkes"
    ###preprocess
    flag_preprocess_dataset = False
    flag_download_data_Amazon = False
    flag_convert_pd = False
    flag_remap_id = False
    flag_build_dataset_user = False
    flag_build_dataset_train = False

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
    #if(dataset_run in dataset_should_process_time_seq_to_startZero):
    #    time_method = "time_parse_start_zero"
    max_seq_len = 50

    #bptt
    bptt_default=20
    bptt_dict = {
        "mimic2": 2,
        "so": 6,
        "ali": 10,
        "lastfm": 8,
        "book_order": 3,        #finantial data
        "hawkes": 2,
        "selfcorrecting": 2,
        "mixture-HMM": 2
    }
    bptt = bptt_dict.get(dataset_run, bptt_default)
    #bptt = 10


    batch_size_default=64
    batch_size_dict = {
        "hawkes": 8,
        "selfcorrecting": 8,
        "mixture-HMM": 8,
        "book_order":8,
        "hawkes5":32
    }
    batch_size = batch_size_dict.get(dataset_run, batch_size_default)


    #
    metrics_K_default=20
    metrics_K_dict = {
        "mimic2":20,
        "ali": 20,
        "so": 10,
        "book_order": 1,
        "lastfm": 50,
        "hawkes": 1,
        "selfcorrecting": 1,
        "mixture-HMM": 1,
        "hawkes5":3
    }
    metrics_K = metrics_K_dict.get(dataset_run, metrics_K_default)

    trackperiod = 100

    #train_num_epochs
    train_num_epochs_default = 10
    train_num_epochs_dict = {
        "mimic2": 100,
        "lastfm": 10,
        "book_order": 10,
        "hawkes": 20,
        "selfcorrecting": 20,
        "mixture-HMM": 20,
        "Beauty": 10,
        "Clothing_Shoes_and_Jewelry": 20
    }
    train_num_epochs = train_num_epochs_dict.get(dataset_run, train_num_epochs_default)
    #train_num_epochs=1

    ##split data
    flag_split_user_seq_OnlyOneTrainFromLast = False
    #if gap > split_sequence_time_gap_max: split sequence
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
        "lastfm": 0.001,
        "book_order": 0.001,
        "so": 0.01,
        "hawkes": 0.001,
        "selfcorrecting": 0.001,
        "mixture-HMM": 0.001

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
        "atrank": "dataset_%s.pkl" % "atrank",
        "SASRec": "dataset_%s.pkl" % "SASRec",
        "rmtpp": "dataset_%s.pkl" % "rmtpp",
        "rmtpp_att": "dataset_%s.pkl" % "rmtpp_att",
        "intensityRNN": "dataset_%s.pkl" % "intensityRNN",
        "AMPP_En": "dataset_train.pkl",
        "AMPP_EnDe": "dataset_train_EnDe.pkl",
        "AMPP_En_SASRec": "dataset_train.pkl",
        "AMPP_EnDe_ATRank": "dataset_train_%s.pkl" % "ATRank",
        "AMPP_En_Seq": "dataset_%s.pkl" % "AMPP_En_Seq",
        "neuraHawkes": "dataset_%s.pkl" % "neuraHawkes",

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
    if(model_run == "intensityRNN"):
        loss_time_method = "gaussian"
        flagMinMax = False
        flagScale = False

    if(model_run == "RNN"):
        model_run = "intensityRNN"
        weight_loss_intensityRNN = "1,0"
        weight_loss_intensityRNN = "0,1"
        loss_time_method="mse"
        flagMinMax = False
        flagScale = False
    if(model_run == "Transformer"):
        model_run = "AMPP_En_Seq"
        weight_loss = "1,0"
        weight_loss = "0,1"
        loss_time_method = "mse"
        flagMinMax = False
        flagScale = False
        #time_method = "time_org"
        time_encode_method = "pos"

    hidden_units = 64





