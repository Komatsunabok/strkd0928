import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample
from dataset.cinic10 import get_cinic10_dataloaders, get_cinic10_dataloaders_sample

def load_dataset(data_folder_dir, dataset='cifar10', batch_size=64, num_workers=8):
    # データロード
    if dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=batch_size, 
                                                            data_folder_dir=data_folder_dir,
                                                            num_workers=num_workers)
    elif dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=batch_size, 
                                                           data_folder_dir=data_folder_dir,
                                                           num_workers=num_workers)
    elif dataset == 'imagenet':
        train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=dataset, batch_size=batch_size,
                                                                          data_folder_dir=data_folder_dir,
                                                                        num_workers=num_workers,
                                                                        multiprocessing_distributed=True)
    elif dataset == 'cinic10':
        train_loader, val_loader = get_cinic10_dataloaders(batch_size=batch_size,data_folder_dir=data_folder_dir,
                                                                        num_workers=num_workers)
    else:
        raise NotImplementedError(dataset)
    
    return train_loader, val_loader