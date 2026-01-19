from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from best_config_GSC_former import Config as GSCTConfig
from best_config_SSC_former import Config as SSCTConfig
from best_config_SHD_former import Config as SHDTConfig
from spikcommder import SpikCommander
from spikingjelly.datasets import padded_sequence_mask

from snn_delays import SnnDelays
import torch
from snn import SNN
import utils
import numpy as np
import random, sys
import torch
from utils import init_logger,build_optimizer
from logging import getLogger
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import copy
import matplotlib.pyplot as plt
from cross_entropy import SoftTargetCrossEntropy

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def calc_loss(config, output, y):
    if config.loss == 'mean':
        m = torch.mean(output, 0)
    elif config.loss == 'max':
        m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count':
        m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(m, y)

        return loss

def calc_loss_nonspike(config, output, y):
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)

        return loss

def calc_metric(config, output, y):
    # mean accuracy over batch
    if config.loss == 'mean': m = torch.mean(output, 0)
    elif config.loss == 'max': m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count': m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

    return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())




def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()

    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(loader)):

            attention_mask = padded_sequence_mask(x_len)
            attention_mask = attention_mask.transpose(0,1).to(device)
            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)
            y = y.to(device)

            output = model(x,attention_mask)

            loss = calc_loss(config, output, y)
            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)


    loss_valid = np.mean(loss_batch)
    metric_valid = np.mean(metric_batch)
    return loss_valid, metric_valid



def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(train_loader)):
            attention_mask = padded_sequence_mask(x_len)
            attention_mask = attention_mask.transpose(0,1).to(device)


            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons) => (512,101,140)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(x,attention_mask)
            loss = calc_loss(config, output, y)


            loss.backward()
            optimizer.step()

            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()

        # best_model_wts = copy.deepcopy(model.state_dict())

        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y, x_len) in enumerate(tqdm(valid_loader)):
                attention_mask = padded_sequence_mask(x_len)
                attention_mask = attention_mask.transpose(0, 1).to(device)

                y = F.one_hot(y, config.n_outputs).float()
                x = x.float().to(device)
                y = y.to(device)

                output = model(x,attention_mask)
                loss = calc_loss(config, output, y)
                metric = calc_metric(config, output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                functional.reset_net(model)

        loss_valid = np.mean(loss_batch)
        metric_valid = np.mean(metric_batch)

        loss_epochs['valid'].append(loss_valid)
        metric_epochs['valid'].append(metric_valid)
        #
        if test_loader:
            loss_test, metric_test = eval_model(config, model, test_loader, device)
        else:
            # could be improved
            loss_test, metric_test = 100, 0
        #
        loss_epochs['test'].append(loss_test)
        metric_epochs['test'].append(metric_test)

        ########################## Logging and Plotting  ##########################


        logger.info(
            f"=====> Epoch {epoch} : Loss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100 * metric_epochs['train'][-1]:.2f}%")
        logger.info(
            f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100 * metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100 * max(metric_epochs['valid'][-1], best_metric_val):.2f}%")
        if test_loader:
            logger.info(
                f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100 * metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100 * max(metric_epochs['test'][-1], best_metric_test):.2f}%")

        checkpoint_dir = os.path.join('./checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(),ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

        if metric_test > best_metric_test:
            best_metric_test = metric_test

    ###### make_plot ######
    train_acc = [x * 100 for x in metric_epochs['train']]
    valid_acc = [x * 100 for x in metric_epochs['valid']]
    test_acc = [x * 100 for x in metric_epochs['test']]

    epochs = range(1, len(train_acc) + 1)
    if config.make_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, '-o', label='Training Accuracy', color='g')
        plt.plot(epochs, valid_acc, '-^', label='Validation Accuracy', color='b')
        plt.plot(epochs, test_acc, '-s', label='Test Accuracy', color='y')

        plt.title('Training, Validation, and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f%%'))
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'Accuracy_{current_time}.png')
        plt.show()




if __name__ == '__main__':


    # config = SHDTConfig()
    # config = GSCTConfig()
    config = SSCTConfig()

    logger = init_logger(config,"training")
    logger.info("Logger is properly initialized and ready to use.")
    logger.info("The GPU is {}".format(config.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = config.gpu
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    print()
    print(f"\n=====> Device = {device} \n\n")

    for hidden_dim in config.n_hidden_neurons_list:
        ''' set random seeds '''
        seed_val = config.seed
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        config.attention_window = attention_window


        logger.info("##############################################\n")
        logger.info("Seed :{}".format(seed_val))
        epochs = config.epochs
        logger.info("The epoch is: {}".format(epochs))
        logger.info("The batch size is : {}".format(config.batch_size))
        logger.info("The backend is: {}".format(config.backend))
        logger.info("The num_heads is: {}".format(config.num_heads))
        logger.info("The attn_window_size is: {}".format(config.attention_window))
        if config.use_aug:
            logger.info("The TN_mask_aug_proba is: {}".format(config.TN_mask_aug_proba))
            logger.info("The time_mask_proportion is: {}".format(config.time_mask_proportion))
            logger.info("The neuron_mask_size is: {}".format(config.neuron_mask_size))

        else:
            logger.info("The augs is None")


        """ dataset """
        if config.dataset == 'shd':
            train_loader, valid_loader = SHD_dataloaders(config)
            test_loader = None
        elif config.dataset == 'ssc':
            train_loader, valid_loader, test_loader = SSC_dataloaders(config)
        elif config.dataset == 'gsc':
            train_loader, valid_loader, test_loader = GSC_dataloaders(config)
        else:
            raise Exception(f'dataset {config.dataset} not implemented')

        config.n_hidden_neurons = hidden_dim
        config.hidden_dims = 4 * hidden_dim

        logger.info("The time_step is: {}".format(config.time_step))
        logger.info("The n_bins is: {}".format(config.n_bins))
        logger.info("The hidden_dim is: {}".format(hidden_dim))
        logger.info("The spike_mode is: {}".format(config.spike_mode))
        logger.info("The block_mode is :{}".format(config.block_type))
        
        model = SpikCommander(config).to(device)

        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        dataset_info = config.dataset
        folder_path = os.path.join('model_structure', dataset_info)
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f'model_structure_{dataset_info}_{formatted_time}.txt')

        with open(filename, 'w') as f:
            print(model, file=f)

        print(f"===> Dataset    = {config.dataset}")
        print(f"===> Model type = {config.model_type}")
        print(f"===> Model size = {utils.count_parameters(model)}\n\n")
        logger.info("Model size:{}".format(utils.count_parameters(model)))
        lr_w = config.lr_w
        logger.info("lr_w: {}".format(lr_w))
        weight_decay = config.weight_decay
        logger.info("weight_decay: {}".format(weight_decay))
        optimizer = build_optimizer(config, model)
        T = config.t_max
        logger.info("T:{}".format(T))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

        train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, epochs)