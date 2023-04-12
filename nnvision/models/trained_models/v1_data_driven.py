import torch
import os

import nnvision
import nnfabrik
from nnfabrik.builder import get_model

# full model key
key = {'model_fn': 'nnvision.models.se_core_point_readout',
  'model_hash': 'f8bcd882c48a55dc6cd6d7afb656f1f9',
  'dataset_fn': 'nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
  'dataset_hash': 'ad53bc33a1a89c8e7c9c38bb9ef82474',
  'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
  'trainer_hash': 'e67767f0f592b14b0bebe7d4a96c442d',
  'seed': 2000}

model_fn = 'nnvision.models.se_core_point_readout'
model_config = {'pad_input': False,
 'stack': -1,
 'depth_separable': True,
 'input_kern': 24,
 'gamma_input': 10,
 'gamma_readout': 0.5,
 'hidden_dilation': 2,
 'hidden_kern': 9,
 'se_reduction': 16,
 'n_se_blocks': 2,
 'hidden_channels': 32}

data_info = {
    "all_sessions": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 458,
        "img_mean": 124.54466,
        "img_std": 70.28,
    },
}
current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, '../../data/model_weights/v1_data_driven/v1_sota_cnn.pth.tar')
state_dict = torch.load(filename)

# load single model
v1_data_driven_sota = get_model(
    model_fn, model_config, seed=10, data_info=data_info, state_dict=state_dict
)


# load ensemble model
from mei.modules import EnsembleModel

ensemble_names = ['sota_cnn_ensemble_model_2.pth.tar',
 'sota_cnn_ensemble_model_3.pth.tar',
 'sota_cnn_ensemble_model_4.pth.tar',
 'sota_cnn_ensemble_model_5.pth.tar',]

base_dir = os.path.dirname(filename)
ensemble_models = []
ensemble_models.append(v1_data_driven_sota)

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

# Ensemble model
v1_data_driven_sota_ensemble_model = EnsembleModel(*ensemble_models)


# load second ensemble model with different seeds (models 6-10)
ensemble_names = ['sota_cnn_ensemble_model_6.pth.tar',
    'sota_cnn_ensemble_model_7.pth.tar',
    'sota_cnn_ensemble_model_8.pth.tar',
    'sota_cnn_ensemble_model_9.pth.tar',
    'sota_cnn_ensemble_model_10.pth.tar',]


ensemble_models = []

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

# Second ensemble model
v1_data_driven_sota_ensemble_model_2 = EnsembleModel(*ensemble_models)
