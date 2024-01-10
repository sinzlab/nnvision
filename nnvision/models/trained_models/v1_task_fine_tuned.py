import torch
import os

import nnvision
import nnfabrik
from nnfabrik.builder import get_model

# full model key
key = {'model_fn': 'nnvision.models.ptrmodels.convnext_core_gauss_readout',
  'model_hash': '4cf506abe227ca05230aae13ffcca3d5',
  'dataset_fn': 'nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
  'dataset_hash': 'a416f090b32e84aad6a87e82d7ddf57d',
  'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
  'trainer_hash': 'e145fd0f8537a754b8972d894ed2820d',
  'seed': 8000}

model_fn = 'nnvision.models.ptrmodels.convnext_core_gauss_readout'
model_config =  {'model_name': 'facebook/convnextv2-atto-1k-224',
  'layer_name': 'convnextv2.encoder.stages.1.layers.0',
  'patch_embedding_stride': None,
  'fine_tune': True,
  'pretrained': True,
  'gamma_readout': 10,
  'final_norm': 'BatchNorm',
  'final_nonlinearity': 'GELU'}

data_info = {
    "all_sessions": {
        "input_dimensions": torch.Size([512, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 458,
        "img_mean": 124.54466,
        "img_std": 70.28,
    },
}
current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, '../../data/model_weights/v1_task_fine_tuned/v1_convnext_1.pth.tar')
state_dict = torch.load(filename)

# load single model
v1_convnext = get_model(
    model_fn, model_config, seed=10, data_info=data_info, state_dict=state_dict
)


# load ensemble model
from mei.modules import EnsembleModel

ensemble_names = ['v1_convnext_2.pth.tar',
 'v1_convnext_3.pth.tar',
 'v1_convnext_4.pth.tar',
 'v1_convnext_5.pth.tar',]

base_dir = os.path.dirname(filename)
ensemble_models = []
ensemble_models.append(v1_convnext)

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

# Ensemble model
v1_convnext_ensemble = EnsembleModel(*ensemble_models)