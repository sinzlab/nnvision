import torch
import os

import nnvision
import nnfabrik
from nnfabrik.builder import get_model

# full model key
key = {'dataset_fn': 'nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
  'dataset_hash': '9ef1991a6c99e7d5af6e2a51c3a537a6',
  'ensemble_hash': '46e04f7a762c4f9dcbb5663f315084c6',
  'model_fn': 'nnvision.models.ptrmodels.task_core_gauss_readout',
  'model_hash': 'ade1c26ff74aef5479499079a219474e',
  'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
  'trainer_hash': 'e67767f0f592b14b0bebe7d4a96c442d',}


model_fn = 'nnvision.models.ptrmodels.task_core_gauss_readout'
model_config =  {'input_channels': 1,
  'model_name': 'resnet50_l2_eps0_1',
  'layer_name': 'layer3.0',
  'pretrained': False,
  'bias': False,
  'final_batchnorm': True,
  'final_nonlinearity': True,
  'momentum': 0.1,
  'fine_tune': False,
  'init_mu_range': 0.4,
  'init_sigma_range': 0.6,
  'readout_bias': True,
  'gamma_readout': 3.0,
  'gauss_type': 'isotropic',
  'elu_offset': -1,
                 }

data_info = {
    "all_sessions": {
        "input_dimensions": torch.Size([64, 1, 100, 100]),
        "input_channels": 1,
        "output_dimension": 1244,
        "img_mean": 124.54466,
        "img_std": 70.28,
    },
}

current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, '../../data/model_weights/v4_task_driven/task_driven_ensemble_model_01.pth.tar')
state_dict = torch.load(filename)

# load single model
v4_task_driven_resnet_model = get_model(
    model_fn, model_config, seed=10, data_info=data_info, state_dict=state_dict
)

# load ensemble model
from mei.modules import EnsembleModel

# fill the list ensemble names with task driven 01 - 10
ensemble_names = ['task_driven_ensemble_model_01.pth.tar',
    'task_driven_ensemble_model_02.pth.tar',
    'task_driven_ensemble_model_03.pth.tar',
    'task_driven_ensemble_model_04.pth.tar',
    'task_driven_ensemble_model_05.pth.tar',
    'task_driven_ensemble_model_06.pth.tar',
    'task_driven_ensemble_model_07.pth.tar',
    'task_driven_ensemble_model_08.pth.tar',
    'task_driven_ensemble_model_09.pth.tar',
    'task_driven_ensemble_model_10.pth.tar',]

base_dir = os.path.dirname(filename)
ensemble_models = []

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

task_driven_ensemble_1 = EnsembleModel(*ensemble_models[:5])
task_driven_ensemble_2 = EnsembleModel(*ensemble_models[5:])