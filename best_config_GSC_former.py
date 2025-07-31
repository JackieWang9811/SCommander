from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'gsc'
    datasets_path = 'Datasets/GSC'
    log_dir = './logs/logging/gsc/'
    log_dir_test = './logs/logging/testing'

    seed = 312  # 312 42 3407 0 10086 114514+-5 3112

    gpu = 1
    model_type = 'spike-temporal-former'
    block_type = 'sequeezeformer'  # 'sequeezeformer'

    distribute = False

    spike_mode = "lif"

    # Spectrogram
    # Number of Frames= 1+ (Total Length of the Audio - Window Size) / Hop_Length
    # Learning delays: hop_length=80; n_mels=140
    window_size = 256 # 256
    # 40+70, 80+140或许是两个组合
    hop_length = 80  # 16=> 500 time steps, 20=>400 time steps, 25=>320 time steps,  40=> 200 time steps, 80 => 100 time steps, 200 => 40 time steps
    # hop_length_list = [800,400,200,160,40,32]  # 16=> 500 time steps, 20=>400 time steps, 25=>320 time steps,  40=> 200 time steps, 80 => 100 time steps, 200 => 40 time steps
    hop_length_list = [32]  # 16=> 500 time steps, 20=>400 time steps, 25=>320 time steps,  40=> 200 time steps, 80 => 100 time steps, 200 => 40 time steps
    # hop_length_list = [80]  # 16=> 500 time steps, 20=>400 time steps, 25=>320 time steps,  40=> 200 time steps, 80 => 100 time steps, 200 => 40 time steps
    n_mels = 140 # 140 => 70

    attention_window = 20 # 4,8,12,24,32 ,跑的顺序，24=》12=》8=》4=》32
    attention_window_list = [8]
    # attention_window_list = [16, 24]
    # attention_window_list = [28, 32]


    depths = 2
    epochs = 300 if spike_mode == "plif" else 300
    # batch_size = 128 if hop_length==20 else 256# 256 => 512
    # if hop_length == 8 and depths == 1:
    #     batch_size = 72
    # elif hop_length == 16 and depths == 1:
    #     batch_size = 208
    # elif hop_length == 16 and depths == 2:
    #     batch_size = 126
    # elif hop_length == 20 and depths == 1:
    #     batch_size = 256
    # elif hop_length == 20 and depths == 2:
    #     batch_size = 128
    # else:
    # batch_size = 248

    batch_size = 256
    # batch_size = 256

    # dropout_l control the first layer
    dropout_l = 0.1
    # dropout_p control the layers in attention
    dropout_p = 0.1

    # MLP_RATIO
    mlp_ratio = 4
    # SPLIT_RATIO
    split_ratio = 1
    ############################
    #        USE Module        #
    ############################
    # 控制不同模块间是否使用BN
    use_norm = False
    # 控制每个模块的第一个输入是否过一个LN
    use_ln = False
    # 控制每个block最后一个输入是否是一个scale
    use_adaptive_scale = True
    # 控制首尾是否加入残差
    use_identity = False
    # 控制每个module之间是否使用LIF
    use_lif = False
    # GSU中是否要使用BN
    use_bn = True
    # 是否使用SepcAug数据增强
    use_aug = True
    # 是否使用dropout
    use_dp = True
    # 是否使用DW的biass
    use_dw_bias = False

    use_global = True
    use_local = False

    ############################
    #          Augment         #
    ############################

    #  SpecAugment #
    mF = 1
    F = 10
    mT = 1
    pS = 0.25

    # SpecAugmenter
    n_time_masks = 2
    time_mask_width =  25
    n_freq_masks =  2
    freq_mask_width =  7


    backend = 'cupy'
    attn_mode = 'v2'
    kernel_size = 31  # 卷积核为255时，92.42% 255=>127=>63=>31,
    bias = True

    # weight_decay = 1e-5
    n_warmup = 0
    # lr_start = 1e-5
    t_max = 40
    lr_w = 2e-3 # 2e-3
    weight_decay = 5e-3 # Default 0.1 => 0.01 => 2e-3 => 5e-3

    n_inputs = n_mels
    n_hidden_neurons_list = [256] # [128, 144, 160, 176, 192, 208, 224, 240, 256]
    n_hidden_neurons = 144
    n_outputs = 20 if dataset == 'shd' else 35
    hidden_dims = mlp_ratio*n_hidden_neurons # 可以试一下768

    num_heads = 16  # 4=> 8=> 16 不增添加网络参数
    # spike_mode_list = ['lif', 'plif']

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 if spike_mode == "plif" else 2.0  # LIF
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    gate_v_threshold = 1.0 # LIF
    alpha = 5.0

    # surrogate_function = surrogate.Sigmoid(alpha=alpha)
    surrogate_function = surrogate.ATan(alpha=alpha)  # FastSigmoid(alpha)
    detach_reset = True


    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adamw'
    optimizer_pos = 'adamw'


    ################################################
    #                    Save                      #
    ################################################
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model_type
    run_info = f'||{dataset}||{depths}depths||{window_size}ms||bins={hop_length}||lr_w={lr_w}||heads={num_heads}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
