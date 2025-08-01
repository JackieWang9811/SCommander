from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'ssc'
    datasets_path = 'Datasets/SSC'
    log_dir = './logs/logging/ssc/'
    log_dir_test = './logs/logging/testing'

    seed = 312  # 312 42 3407 0 10086 114514+-5 3112
    gpu = 1
    model_type = 'spikcommander'
    block_type = 'mstasa'

    distribute = False

    spike_mode = "lif"
    depths = 1
    time_step = 10
    n_bins = 5
    n_warmup  = 10
    epochs = 300
    attention_window = 20

    batch_size = 256
    # dropout_l control the first layer
    dropout_l = 0.1
    # dropout_p control the layers in attention
    dropout_p = 0.1
    # MLP_RATIO
    mlp_ratio = 4
    #
    split_ratio = 1

    ############################
    #        USE Module        #
    ############################
    use_bn = True
    use_aug = True
    use_dp = True
    use_dw_bias = False



    ############################
    #          Augment         #
    ############################
    #    TimeNeurons_mask Aug  #
    TN_mask_aug_proba = 0.5
    time_mask_proportion = 0.1
    neuron_mask_size= 10

    backend = 'cupy'
    attn_mode = 'v2'
    kernel_size = 31
    bias = True

    t_max = 40
    lr_w = 0.005
    weight_decay = 0.01 # Default 1e-2 => 5e-4
    n_inputs = 700//n_bins
    n_hidden_neurons_list = [256] # [128, 144, 160, 176, 192, 208, 224, 240, 256]
    n_outputs = 20 if dataset == 'shd' else 35

    num_heads = 16
    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 if spike_mode == "plif" else 2.0  # LIF
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    gate_v_threshold = 1.0  # LIF
    alpha = 5.0
    # surrogate_function = surrogate.Sigmoid(alpha=alpha)
    surrogate_function = surrogate.ATan(alpha=alpha)  # FastSigmoid(alpha)
    detach_reset = True
    init_w_method = 'kaiming_uniform'
    max_len = 126
    use_padding = False
    norm_type = "bn"

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
    run_info = f'||{dataset}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}||heads={num_heads}||gate_v={gate_v_threshold}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
