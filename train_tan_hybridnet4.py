#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import argparse
from math import ceil
import torch
import torchvision
import torchvision.transforms as transforms
from hybridnet4 import HybridNet
import torch.nn as nn
import torch.optim as optim
from opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
from opacus.validators import ModuleValidator
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Subset
from opacus import GradSampleModule
# from EMA import EMA
from models.EMA_without_class import create_ema, update
import pickle
from models.augmented_grad_samplers import AugmentationMultiplicity
from math import ceil
from test import (init_distributed_mode,initialize_exp,bool_flag,accuracy,get_noise_from_bs,get_epochs_from_bs,print_params,)
import json
from models.prepare_models import prepare_data_cifar, prepare_augmult_cifar
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")

train_acc = []

def get_activations(model, input_tensor):
    activations = {}
    
    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.Linear)):  # Adjust based on your interest
            layer_name = module.__class__.__name__ + "_" + str(id(module))
            activations[layer_name] = output.detach()
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations

# def save_activations(activations, nb_steps):
#     save_dir = f'activations_dp_iter_40'
#     os.makedirs(save_dir, exist_ok=True)
#     for i, activation in activations.items():
#         sanitized_name = i.replace('.', '_')
#         file_name = os.path.join(save_dir, f"{sanitized_name}_actvs_dp_iter_{nb_steps}.npy")
#         np.save(file_name, activation.cpu().numpy())


def save_activations(activations, epoch):
    save_dir3 = f'actvs_{epoch+1}'
    os.makedirs(save_dir3, exist_ok=True)
    for i, activation in activations.items():
        sanitized_name = i.replace('.', '_')
        file_path3 = os.path.join(save_dir3, f"{sanitized_name}_actvs_epoch_{epoch+1}.npy")
        np.save(file_path3, activation.cpu().numpy())
    print(f"Activations saved for epoch {epoch+1}")


def train(
    model,
    ema,
    train_loader,
    optimizer,
    epoch,
    max_nb_steps,
    target_iter,
    device,
    privacy_engine,
    K,
    logger,
    losses,
    train_acc,
    epsilons,
    grad_sample_gradients_norms_per_epoch,
    test_loader,
    is_main_worker,
    args,
    norms2_before_sigma,
    nb_steps,
    train_accuracy
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """
    # nb_steps = nb_steps
    train_accuracy = []
    model.train()
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    if is_main_worker:print(f"steps_per_epoch:{steps_per_epoch}")
    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    nb_examples_epoch = 0
    max_physical_batch_size_with_augnentation = (args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K)
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            nb_examples_epoch+=len(images)
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            target = target.to(device)
            assert K == args.transform
            l = len(images)
            ##Using Augmentation multiplicity
            if K:
                images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
                target = torch.repeat_interleave(target, repeats=K, dim=0)
                transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),transforms.RandomHorizontalFlip(p=0.5),])
                images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
                assert len(images) == args.transform * l

            # compute output
            # print(f"Input images shape: {images.shape}")
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)

            loss.backward()
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch

            ## Logging gradient statistics on the main worker
            if is_main_worker:
                per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s

        
            optimizer.step()
            if is_updated:
                nb_steps += 1  # ?
                if ema:
                    update(model, ema, nb_steps)
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    grad_sample_gradients_norms_per_epoch.append(np.mean(grad_sample_norms))
                    losses_epoch, train_acc_epoch = [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        m2 = max(np.mean(norms2_before_sigma)-1/args.batch_size,0)
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "nb_steps":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "grad_sample_gradients_norms": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    #"norms2_before_sigma":list(norms2_before_sigma),
                                   # "grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                }
                            )
                        )
                        train_accuracy.append(np.mean(train_acc[-args.freq_log :]))

                        try:
                            with open(f'train_acc_hybridnet4.0+{epoch}.json' , 'w') as file:
                                json.dump(train_accuracy, file)
                            print('here')
                        except:
                            with open(f'train_acc_hybridnet4.0+{epoch}.json', 'r') as file:
                                data_list = json.load(file)
                            data_list.extend(train_accuracy)

                            with open(f'train_acc_hybridnet4.0+{epoch}.json' , 'w') as file:
                                json.dump(data_list, file)
                                
                        norms2_before_sigma=[]
                        grad_sample_norms = []
                    if nb_steps % args.freq_log_val == 0:
                        test_acc_ema, train_acc_ema, test_loss_ema = (
                            test(ema, test_loader, train_loader, device)
                            if ema
                            else test(model, test_loader, train_loader, device)
                        )
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "test_acc_ema": test_acc_ema,
                                    "train_acc_ema": train_acc_ema,
                                    "test_loss_ema": test_loss_ema
                                }
                            )
                        )
                        try:
                            with open(f'train_ema_hybridnet4.0+{epoch}.json' , 'w') as file:
                                json.dump(train_acc_ema, file)
                            print('here')
                        except:
                            with open(f'train_ema_hybridnet4.0+{epoch}.json', 'r') as file:
                                data_list = json.load(file)
                            data_list.extend(train_acc_ema)

                            with open(f'train_ema_hybridnet4.0+{epoch}.json' , 'w') as file:
                                json.dump(data_list, file)
                nb_examples_epoch=0
                if nb_steps >= max_nb_steps:
                    break

        # Check if the current epoch matches the target epoch for saving weights
        if epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3 or epoch == 4:
            save_dir = f'weights_epoch_{epoch+1}'
            os.makedirs(save_dir, exist_ok=True)
            for name, parameter in model.named_parameters():
                if parameter is not None:
                    sanitized_name = name.replace('.', '_')
                    file_path = os.path.join(save_dir, f"{sanitized_name}_weights_epoch_{epoch+1}.npy")
                    np.save(file_path, parameter.detach().cpu().numpy())
            print(f"Weights saved for epoch {epoch+1}")

            save_dir2 = f'grads_epoch_{epoch+1}'
            os.makedirs(save_dir2, exist_ok=True) 
            # Iterate over all named parameters to access and save their weights
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    # Replace dots in layer names with underscores to avoid file naming issues
                    sanitized_name = name.replace('.', '_')
                    file_path2 = os.path.join(save_dir2, f"{sanitized_name}_gradients_epoch_{epoch+1}.npy")
                    # Save the gradient as a NumPy array
                    np.save(file_path2, parameter.grad.detach().cpu().numpy())
            print(f"Gradients saved for epoch {epoch+1}")
            
            activations = get_activations(model, images.to(device))
            save_activations(activations, epoch)


        epsilon = privacy_engine.get_epsilon(args.delta)
        if is_main_worker:
            epsilons.append(epsilon)
        return nb_steps, norms2_before_sigma, {
            'train_loss': np.mean(losses_epoch) if losses_epoch else 0,
            'train_accuracy': np.mean(train_acc_epoch) if train_acc_epoch else 0
        }


def test(model, test_loader, train_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []
    train_top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_accuracy = np.mean(test_top1_acc)

    with torch.no_grad():
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # losses.append(loss.item())
            train_top1_acc.append(acc)
    train_top1_avg = np.mean(train_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")

    # Calculate average test loss and accuracy
    test_loss = np.mean(losses)
    test_accuracy = np.mean(test_top1_acc)
    train_top1_avg = np.mean(train_top1_acc)

    return test_accuracy, train_top1_avg, test_loss

# def plot_metrics(metrics):
#     epochs = range(len(metrics['train_loss']))

#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(2, 2, 1)
#     plt.plot(epochs, metrics['train_loss'], label='Training Loss')
#     plt.title('DP Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(2, 2, 2)
#     plt.plot(epochs, metrics['test_loss'], label='Test Loss')
#     plt.title('DP Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(2, 2, 3)
#     plt.plot(epochs, metrics['train_accuracy'], label='Training Accuracy')
#     plt.title('DP Training Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()

#     plt.subplot(2, 2, 4)
#     plt.plot(epochs, metrics['test_accuracy'], label='Test Accuracy')
#     plt.title('DP Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()


#     plt.tight_layout()
#     plt.savefig("tan_dp_plot.png")



def plot_metrics(metrics):
    epochs = range(len(metrics['train_loss']))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Training Loss')
    plt.plot(epochs, metrics['test_loss'], label='Test Loss')
    plt.title('DP Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, metrics['test_accuracy'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("tan_dp_plot_hybridnet4.png")



def main():  ## for non poisson, divide bs by world size

    #args commented below and pasted before device
    args = parse_args()
    target_iter = args.target_iter

    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model = HybridNet().to(device)


    # args = parse_args()
    # init_distributed_mode(args)#Handle single and multi-GPU / multi-node]
    init_distributed_mode(args)
    logger = initialize_exp(args)
    model = HybridNet().to(device)
   
    #uncomment below if needed
    # model = ConvNet(BasicBlock,args.WRN_depth,args.WRN_k,args.nb_groups)
    # model.cuda()
    print_params(model)
    if args.multi_gpu:
        print("using multi GPU DPDDP")
        model = DPDDP(model)
    rank = args.global_rank
    is_main_worker = rank == 0
    weights = model.module.parameters() if args.multi_gpu else model.parameters()
    # Creating the datasets and dataloader for cifar10
    train_dataset,train_loader, test_loader = prepare_data_cifar(args.data_root,args.batch_size,args.proportion)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(weights, lr=args.lr, momentum=args.momentum)
    # Creating the privacy engine
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)
    sigma = get_noise_from_bs(args.batch_size, args.ref_noise, args.ref_B)

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )
    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult_cifar(model,args.transform)
    ema = None
    # we create a shadow model
    print("shadowing de model with EMA")
    ema = create_ema(model)
    train_acc,test_acc,epsilons,losses,top1_accs,grad_sample_gradients_norms_per_step = (0, 0, [], [], [], [])
    norms2_before_sigma = []

    E = get_epochs_from_bs(args.batch_size, args.ref_nb_steps, len(train_dataset))
    if is_main_worker: print(f"E:{E},sigma:{sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{sigma}, EPOCHS:{E}")
    nb_steps = 0



    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    train_accuracy = []


    for epoch in range(E):
        if nb_steps >= args.ref_nb_steps:
            break
        nb_steps, norms2_before_sigma, train_metrics = train(
            model,
            ema,
            train_loader,
            optimizer,
            epoch,
            args.ref_nb_steps,
            target_iter,
            device, #device
            # rank,
            privacy_engine,
            args.transform,
            logger,
            losses,
            top1_accs,
            epsilons,
            grad_sample_gradients_norms_per_step,
            test_loader,
            is_main_worker,
            args,
            norms2_before_sigma,
            nb_steps,
            train_accuracy
        )

        metrics['train_loss'].append(train_metrics['train_loss'])
        metrics['train_accuracy'].append(train_metrics['train_accuracy'])
        test_acc, train_acc, test_loss = test(model, test_loader, train_loader, device)
        metrics['test_loss'].append(test_loss)
        metrics['test_accuracy'].append(test_acc)

        if is_main_worker:
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f},average gradient norm:{grad_sample_gradients_norms_per_step[-1]:.2f}")
    if is_main_worker:


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        
        test_acc, train_acc, test_loss = test(ema, test_loader, train_loader, device)
        print(f"train_acc (EMA):{train_acc:.2f}, test_acc (EMA):{test_acc:.2f}, epsilon:{epsilons[-1]:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_ema": train_acc,
                    "final_test_acc_ema": test_acc,
                    "final_test_loss_ema": test_loss,
                    "final_epsilon": epsilons[-1],
                    "avergage_grad_sample_gradients_norms": np.mean(
                        grad_sample_gradients_norms_per_step)
                }))
        test_acc, train_acc, test_loss = test(model, test_loader, train_loader, device)
        print(f"final test acc of non ema model:{test_acc:.2f}, final train acc of non ema model:{train_acc:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_non_ema": train_acc,
                    "final_test_acc_non_ema": test_acc,
                    "final_test_loss_non_ema": test_loss,
                }))
        plot_metrics(metrics)

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument("--batch_size",default=256,type=int,help="Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$")
    parser.add_argument("--max_physical_batch_size",default=128,type=int,help="max_physical_batch_size for BatchMemoryManager",)
    parser.add_argument("--WRN_depth",default=4,type=int)
    parser.add_argument("--WRN_k",default=3,type=int,help="k of resnet block",)
    
    #for activations
    parser.add_argument('--actvs', type=int, default=None, help='Specify the epoch for generating activations for all layers')
    
    #for gradients
    # parser.add_argument("--target_iter", nargs='+', type=int, help="Iterations at which to save weights and gradients")
    # parser.add_argument("--target_iter", type=lambda x: [int(i) for i in x.split(',')], help="Iterations at which to save weights and gradients")
    parser.add_argument('--target_iter', type=int, default=None, help='Specify the iteration number for generating activations/weights/gradients for all layers')

    #for saving targets at specified epoch
    parser.add_argument('--save_epoch', type=int, default=None, help='Epoch number at which to save model weights')


    parser.add_argument("--lr","--learning_rate",default=4,type=float,metavar="LR",help="initial learning rate",dest="lr",)

    parser.add_argument("--momentum",default=0,type=float,help="SGD momentum",)
    parser.add_argument("--experiment",default=0,type=int,help="seed for initializing training. ")
    parser.add_argument("-c","--max_per_sample_grad_norm",type=float,default=1,metavar="C",help="Clip per-sample gradients to this norm (default 1.0)",)
    parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta (default: 1e-5)",)
    parser.add_argument("--ref_noise",type=float,default=3,help="reference noise used with reference batch size and number of steps to create our physical constant",)
    parser.add_argument("--ref_B",type=int,default=4096,help="reference batch size used with reference noise and number of steps to create our physical constant",)
    parser.add_argument("--nb_groups",type=int,default=16,help="number of groups for the group norms",)
    parser.add_argument("--ref_nb_steps",default=2500,type=int,help="reference number of steps used with reference noise and batch size to create our physical constant",)
    parser.add_argument("--data_root",type=str,default="./data",help="Where CIFAR10 is/will be stored",)
    parser.add_argument("--dump_path",type=str,default="",help="Where results will be stored",)
    parser.add_argument("--transform",type=int,default=0,help="using augmentation multiplicity",)

    parser.add_argument("--freq_log", type=int, default=20, help="every each freq_log steps, we log",)

    parser.add_argument("--freq_log_val",type=int,default=100,help="every each freq_log steps, we log val and ema acc",)

    parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling",)


    parser.add_argument("--proportion",default=1,type=float,help="proportion of the training set to use for training",)

    parser.add_argument("--exp_name", type=str, default="bypass")

    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--order1", type=int, default=0)
    parser.add_argument("--order2", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()