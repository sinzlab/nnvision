import cnexp
from cnexp.dataset.cifar import load_cifar10
from cnexp.dataset.dataloader import make_dataloaders, make_dataloader


def cifar10_contrastive_loader(path='/data/', batch_size=1000, seed=None):
    dataset_dict = load_cifar10(root=path)
    dataloaders = make_dataloaders(dataset_dict, batch_size=batch_size)

    return dataloaders
