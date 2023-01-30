import torch
import os

import nnvision
import nnfabrik
from nnfabrik.builder import get_model

# full model key
key = {
    "model_fn": "nnvision.models.models.se_core_shared_multihead_attention",
    "model_hash": "bcf9ce3e3869d5ac0cf83e7fcef54e58",
    "dataset_fn": "nnvision.datasets.monkey_loaders.monkey_static_loader_combined",
    "dataset_hash": "9ef1991a6c99e7d5af6e2a51c3a537a6",
    "trainer_fn": "nnvision.training.trainers.nnvision_trainer",
    "trainer_hash": "e67767f0f592b14b0bebe7d4a96c442d",
    "seed": 9000,
}

model_fn = "nnvision.models.models.se_core_shared_multihead_attention"
model_config = {
    "pad_input": False,
    "gamma_input": 10,
    "layers": 5,
    "depth_separable": True,
    "n_se_blocks": 0,
    "stack": -1,
    "input_kern": 9,
    "hidden_kern": 5,
    "hidden_channels": 96,
    "hidden_dilation": 1,
    "linear": False,
    "use_pos_enc": True,
    "dropout_pos": 0.1,
    "final_batch_norm": True,
    "final_nonlinearity": True,
    "key_embedding": True,
    "value_embedding": True,
    "layer_norm": False,
    "scale": True,
    "learned_pos": False,
    "embed_out_dim": 128,
    "gamma_embedding": 0,
    "gamma_query": 1,
    "gamma_features": 1,
    "heads": 1,
    "temperature": [True, 1],
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
filename = os.path.join(current_dir, '../../data/model_weights/data_driven/v4_multihead_attention_SOTA.pth.tar')
state_dict = torch.load(filename)

# load single model
v4_multihead_attention_model = get_model(
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
ensemble_models.append(v4_multihead_attention_model)

for f in ensemble_names:
    ensemble_filename = os.path.join(base_dir, f)
    ensemble_state_dict = torch.load(ensemble_filename)
    ensemble_model = get_model(
        model_fn, model_config, seed=10, data_info=data_info, state_dict=ensemble_state_dict
    )
    ensemble_models.append(ensemble_model)

v4_multihead_attention_ensemble_model = EnsembleModel(*ensemble_models)