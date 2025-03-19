import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
from math import ceil


def get_epochs_from_bs(B, ref_nb_steps, size_dataset):
    """
    output the approximate number of epochs necessary to keep our "physical constant" eta constant.
    We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
    """
    return(ceil(ref_nb_steps*B/size_dataset))

def get_activations(model, data_loader, epoch, save_dir):
    model.eval()  # Set the model to evaluation mode
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu().numpy())
        return hook

    # Register hooks to collect activations
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Process the data for the specified epoch
    for i, data in enumerate(data_loader, start=1):
        if i > epoch:
            break
        inputs, _ = data
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save the activations in separate files
    os.makedirs(save_dir, exist_ok=True)
    for name, acts in activations.items():
        filename = os.path.join(save_dir, f'{name}_epoch_{epoch}.npy')
        np.save(filename, np.concatenate(acts, axis=0))


class AugmentedDataset(Dataset):
    def __init__(self, dataset, multiplicity=16):
        self.dataset = dataset
        self.multiplicity = multiplicity

    def __len__(self):
        # Return the total count of data points, multiplied by the augmentation multiplicity
        return len(self.dataset) * self.multiplicity

    def __getitem__(self, idx):
        # Determine the original index of the image and the augmentation instance
        original_idx = idx // self.multiplicity

        # Apply the transformation each time this method is called
        # Since transforms include random operations, it will result in different augmentations
        image, label = self.dataset[original_idx]

        return image, label
    


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_datasets= AugmentedDataset(train_dataset, multiplicity=16)

    batch_size = batch_size

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
