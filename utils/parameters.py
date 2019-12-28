class Parameters():

    # general parameters
    latent_size = 13
    num_epochs = 12
    learning_rate = 0.001
    batch_size = 128

    # for decoding
    temperature = 1.0
    gen_length = 40
    gen_min_length = 4
    gen_num = 1000

    # encoder
    rnn_layers = 2
    encoder_hidden = 800

    # highway networks
    keep_rate = 1.0
    highway_lc = 2
    highway_ls = 1600

    # decoder
    decoder_hidden = 800
    decoder_rnn_layers = 2
    dec_keep_rate = 0.62

    # data
    embed_size = 353
    sent_max_size = 100
    dataset = "movie"          # tweet, movie
    debug = False
    vocab_drop = 7

    # use pretrained word2vec embeddings
    pre_trained_embed = True
    fine_tune_embed = True

    # technical parameters
    mode = "train"            # train, generate stega text, generate text
    name = "2-800"
    LOG_DIR = "model_logs/" + "VAE_LSTM-LSTM/" + dataset + "/"
    GEN_DIR = "generate/" + "VAE_LSTM-LSTM/" + dataset + "/"


